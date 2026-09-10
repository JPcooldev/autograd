/*
 * Checkpoints: named_parameters, AGCK save/load, strict vs non-strict.
 *
 * - Linear named_parameters are weight and bias
 * - Module dotted child names
 * - BatchNorm state_dict includes running buffers
 * - RNN cell names (weight_ih_l0, …)
 * - in-memory roundtrip copies values and keeps tensor identity
 * - loaded Linear matches source forward
 * - shape mismatch throws; failed load does not mutate matching tensors
 * - missing / unexpected keys throw in strict mode
 * - file save/load including BatchNorm buffers
 * - non-strict skips extra and missing keys
 * - bad magic header throws
 */

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"
#include "../../src/nn/layers/linear.h"
#include "../../src/nn/layers/normalization.h"
#include "../../src/nn/layers/rnn.h"
#include "../../src/nn/module.h"
#include "../../src/nn/serialize.h"

namespace {

class MLP : public nn::Module<float> {
public:
    nn::Linear<float> fc1{4, 3};
    nn::Linear<float> fc2{3, 2, false};

    MLP() {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }
};

bool same_values(const tensor::Tensor<float>& a, const tensor::Tensor<float>& b) {
    if (a.shape() != b.shape())
        return false;
    const auto va = a.to_vector();
    const auto vb = b.to_vector();
    for (size_t i = 0; i < va.size(); ++i) {
        if (va[i] != vb[i])
            return false;
    }
    return true;
}

} // namespace

TEST_CASE("Linear named_parameters uses weight and bias") {
    nn::Linear<float> lin(3, 2);
    auto named = lin.named_parameters();
    REQUIRE(named.size() == 2);
    CHECK(named[0].first == "weight");
    CHECK(named[0].second == &lin.weight);
    CHECK(named[1].first == "bias");
    CHECK(named[1].second == &(*lin.bias));
}

TEST_CASE("Module named_parameters uses dotted child names") {
    MLP model;
    auto named = model.named_parameters();
    REQUIRE(named.size() == 3);
    CHECK(named[0].first == "fc1.weight");
    CHECK(named[0].second == &model.fc1.weight);
    CHECK(named[1].first == "fc1.bias");
    CHECK(named[2].first == "fc2.weight");
    CHECK(model.parameters().size() == 3);
}

TEST_CASE("BatchNorm state_dict includes running buffers") {
    nn::BatchNorm<float> bn(4);
    auto sd = bn.state_dict();
    REQUIRE(sd.size() == 4);
    CHECK(sd[0].first == "weight");
    CHECK(sd[1].first == "bias");
    CHECK(sd[2].first == "running_mean");
    CHECK(sd[3].first == "running_var");
    CHECK(bn.named_parameters().size() == 2);
    CHECK(bn.named_buffers().size() == 2);
}

TEST_CASE("RNN named_parameters follow PyTorch cell names") {
    nn::RNN<float> rnn(3, 5, 1);
    auto named = rnn.named_parameters();
    REQUIRE(named.size() == 4);
    CHECK(named[0].first == "weight_ih_l0");
    CHECK(named[1].first == "weight_hh_l0");
    CHECK(named[2].first == "bias_ih_l0");
    CHECK(named[3].first == "bias_hh_l0");
}

TEST_CASE("save/load roundtrip copies values and keeps tensor identity") {
    MLP src;
    src.fc1.weight.data()[0] = 42.f;
    src.fc1.bias->data()[0] = -7.f;
    src.fc2.weight.data()[0] = 3.5f;

    std::ostringstream oss(std::ios::binary);
    nn::write_checkpoint(oss, nn::snapshot(src));

    MLP dest;
    float* weight_ptr = dest.fc1.weight.data().data();
    std::istringstream iss(oss.str(), std::ios::binary);
    nn::load_state_dict(dest, nn::read_checkpoint<float>(iss));

    CHECK(dest.fc1.weight.data().data() == weight_ptr);
    CHECK(dest.fc1.weight.requires_grad());
    CHECK(same_values(dest.fc1.weight, src.fc1.weight));
    CHECK(same_values(*dest.fc1.bias, *src.fc1.bias));
    CHECK(same_values(dest.fc2.weight, src.fc2.weight));
}

TEST_CASE("loaded Linear matches source forward") {
    nn::Linear<float> src(3, 2);
    tensor::Tensor<float> x({1, 3}, {0.5f, 1.5f, -0.5f}, false);
    auto y_src = src.forward(x);

    std::ostringstream oss(std::ios::binary);
    nn::write_checkpoint(oss, nn::snapshot(src));

    nn::Linear<float> dest(3, 2);
    std::istringstream iss(oss.str(), std::ios::binary);
    nn::load_state_dict(dest, nn::read_checkpoint<float>(iss));
    CHECK(same_values(dest.forward(x), y_src));
}

TEST_CASE("load rejects a shape mismatch for the same key") {
    MLP src;
    std::ostringstream oss(std::ios::binary);
    nn::write_checkpoint(oss, nn::snapshot(src));

    class OtherMLP : public nn::Module<float> {
    public:
        nn::Linear<float> fc1{8, 3};
        nn::Linear<float> fc2{3, 2, false};
        OtherMLP() {
            register_module("fc1", fc1);
            register_module("fc2", fc2);
        }
    } dest;

    std::istringstream iss(oss.str(), std::ios::binary);
    auto records = nn::read_checkpoint<float>(iss);
    CHECK_THROWS_WITH_AS(
        nn::load_state_dict(dest, records),
        doctest::Contains("size mismatch for 'fc1.weight'"),
        std::invalid_argument);
}

TEST_CASE("failed load does not mutate matching tensors") {
    MLP src;
    src.fc2.weight.data()[0] = 99.f;
    std::ostringstream oss(std::ios::binary);
    nn::write_checkpoint(oss, nn::snapshot(src));

    class OtherMLP : public nn::Module<float> {
    public:
        nn::Linear<float> fc1{8, 3};
        nn::Linear<float> fc2{3, 2, false};
        OtherMLP() {
            register_module("fc1", fc1);
            register_module("fc2", fc2);
        }
    } dest;
    const float before = dest.fc2.weight.data()[0];
    std::istringstream iss(oss.str(), std::ios::binary);
    CHECK_THROWS_AS(
        nn::load_state_dict(dest, nn::read_checkpoint<float>(iss)),
        std::invalid_argument);
    CHECK(dest.fc2.weight.data()[0] == before);
}

TEST_CASE("load rejects missing and unexpected keys") {
    MLP src;
    std::ostringstream oss(std::ios::binary);
    nn::write_checkpoint(oss, nn::snapshot(src));

    nn::Linear<float> dest(4, 3);
    std::istringstream iss(oss.str(), std::ios::binary);
    CHECK_THROWS_WITH_AS(
        nn::load_state_dict(dest, nn::read_checkpoint<float>(iss)),
        doctest::Contains("Error(s) in loading state_dict"),
        std::invalid_argument);
}

TEST_CASE("file save/load roundtrip including BatchNorm buffers") {
    nn::BatchNorm<float> src(3);
    src.weight.data() = {1.5f, 2.5f, 3.5f};
    src.running_mean.data() = {0.1f, 0.2f, 0.3f};
    src.running_var.data() = {1.1f, 1.2f, 1.3f};

    const std::string path = "build/autograd_serialize_test.agck";
    nn::save(src, path);

    nn::BatchNorm<float> dest(3);
    nn::load(dest, path);
    CHECK(same_values(dest.weight, src.weight));
    CHECK(same_values(dest.bias, src.bias));
    CHECK(same_values(dest.running_mean, src.running_mean));
    CHECK(same_values(dest.running_var, src.running_var));
}

TEST_CASE("load_state_dict non-strict skips extra and missing keys") {
    MLP src;
    src.fc1.weight.data()[0] = 3.25f;
    auto recs = nn::snapshot(src);
    recs.push_back({"nope", {1}, tensor::Dtype::Float32, true, false, {1.f}});

    MLP dest;
    dest.fc1.weight.data()[0] = 0.f;
    nn::load_state_dict(dest, recs, false);
    CHECK(dest.fc1.weight.data()[0] == doctest::Approx(3.25f));
}

TEST_CASE("read_checkpoint rejects a bad magic header") {
    std::istringstream iss(std::string("NOPE"), std::ios::binary);
    CHECK_THROWS_AS(nn::read_checkpoint<float>(iss), std::invalid_argument);
}
