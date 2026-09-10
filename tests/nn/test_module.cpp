/*
 * Module: parameter counting, registration rules, and zero_grad.
 *
 * - num_parameters over Linear children
 * - num_parameters includes own register_parameter tensors
 * - empty Module is 0
 * - duplicate register_module name throws
 * - register_parameter rejects requires_grad=false
 * - register_buffer is in named_buffers / state_dict, not parameters()
 * - zero_grad clears child parameter grads
 * - register_parameter rejects a name already used by a module
 */

#include <stdexcept>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"
#include "../../src/nn/layers/linear.h"
#include "../../src/nn/module.h"

TEST_CASE("Module num_parameters sums numel over registered sub-modules") {
    class MLP : public nn::Module<float> {
    public:
        nn::Linear<float> fc1{10, 5};
        nn::Linear<float> fc2{5, 2, false};

        MLP() {
            register_module("fc1", fc1);
            register_module("fc2", fc2);
        }
    };

    MLP model;
    // fc1: weight {5, 10} + bias {5} = 55; fc2: weight {2, 5} = 10
    CHECK(model.num_parameters() == 65);
}

TEST_CASE("Module num_parameters includes own registered tensors") {
    class M : public nn::Module<float> {
    public:
        tensor::Tensor<float> table{{3, 4}, true};

        M() { register_parameter("table", table); }
    };

    M model;
    CHECK(model.num_parameters() == 12);
}

TEST_CASE("empty Module reports zero parameters") {
    nn::Module<float> model;
    CHECK(model.num_parameters() == 0);
}

TEST_CASE("register_module rejects a duplicate name") {
    class M : public nn::Module<float> {
    public:
        nn::Linear<float> a{2, 2};
        nn::Linear<float> b{2, 2};
        M() {
            register_module("fc", a);
            CHECK_THROWS_AS(register_module("fc", b), std::invalid_argument);
        }
    };
    M model;
}

TEST_CASE("register_parameter rejects a tensor without requires_grad") {
    class M : public nn::Module<float> {
    public:
        tensor::Tensor<float> table{{3, 4}, false};
        M() {
            CHECK_THROWS_AS(register_parameter("table", table), std::invalid_argument);
        }
    };
    M model;
}

TEST_CASE("register_buffer appears in named_buffers not parameters") {
    class M : public nn::Module<float> {
    public:
        tensor::Tensor<float> stats{{2}, false};

        M() { register_buffer("stats", stats); }
    };

    M model;
    CHECK(model.parameters().empty());
    REQUIRE(model.named_buffers().size() == 1);
    CHECK(model.named_buffers()[0].first == "stats");
    CHECK(model.state_dict().size() == 1);
}

TEST_CASE("Module zero_grad clears parameter grads") {
    class M : public nn::Module<float> {
    public:
        nn::Linear<float> fc{2, 2, false};
        M() { register_module("fc", fc); }
    };
    M model;
    tensor::Tensor<float> x({1, 2}, std::vector<float>{1.f, 2.f}, true);
    model.fc.forward(x).sum().backward();
    REQUIRE(model.fc.weight.grad() != nullptr);
    model.zero_grad();
    CHECK(model.fc.weight.grad() == nullptr);
}

TEST_CASE("register_parameter rejects a name already used by a module") {
    class M : public nn::Module<float> {
    public:
        nn::Linear<float> fc{2, 2};
        tensor::Tensor<float> extra{{2}, true};
        M() {
            register_module("fc", fc);
            CHECK_THROWS_AS(register_parameter("fc", extra), std::invalid_argument);
        }
    };
    M model;
}
