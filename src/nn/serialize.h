#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../logging/logger.h"
#include "../tensor/dtype.h"
#include "layers/layer.h"

namespace nn {

// ---------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------
// Binary format (little-endian) written by save() / read by load():
//
//   magic        4 bytes     "AGCK"
//   version      u32         1
//   n_tensors    u32
//   for each tensor:
//     name_len   u32
//     name       name_len bytes (UTF-8, no NUL)
//     dtype      u32         0=Float32 1=Float64 2=Int32 3=Int64
//     requires_grad u8
//     is_buffer  u8          0=parameter 1=buffer
//     reserved   u16         0
//     rank       u32
//     shape      i64[rank]
//     numel      i64         must equal product(shape)
//     values     T[numel]    packed row-major logical order
//
// Values are always packed (contiguous C-order). Views/strides are not stored;
// load copies into the already-constructed model via Tensor::copy_ so object
// identity (optimizer pointers, GradStorage) is preserved.
//
// load() matches records by name and, in strict mode (default), requires the
// checkpoint keys to equal the model's state_dict keys with the same shape,
// dtype, requires_grad, and parameter-vs-buffer kind.

/**
 * One named tensor as stored in a checkpoint: structure plus packed values.
 *
 * `values` is row-major logical order (what `Tensor::to_vector()` returns),
 * not the possibly-strided backing buffer.
 */
template <typename T>
struct TensorRecord {
    std::string name;
    std::vector<int64_t> shape;
    tensor::Dtype dtype;
    bool requires_grad;
    bool is_buffer;
    std::vector<T> values;
};

namespace checkpoint_detail {

constexpr char kMagic[4] = {'A', 'G', 'C', 'K'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kMaxNameLen = 65536;
constexpr uint32_t kMaxRank = 16;

/**
 * Format a shape vector as `{d0, d1, ...}` for error messages.
 *
 * @param shape Dimensions to print.
 * @return Brace-delimited comma-separated dims.
 */
inline std::string format_shape(const std::vector<int64_t>& shape) {
    std::string s = "{";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i)
            s += ", ";
        s += std::to_string(shape[i]);
    }
    s += "}";
    return s;
}

/**
 * Map a runtime dtype tag to a short name.
 *
 * @param dtype Tensor dtype.
 * @return `"float32"`, `"float64"`, `"int32"`, or `"int64"`.
 */
inline const char* dtype_name(tensor::Dtype dtype) {
    switch (dtype) {
        case tensor::Dtype::Float32: return "float32";
        case tensor::Dtype::Float64: return "float64";
        case tensor::Dtype::Int32: return "int32";
        case tensor::Dtype::Int64: return "int64";
    }
    return "unknown";
}

/**
 * Runtime dtype tag for storage type T.
 *
 * @return Dtype corresponding to T.
 *
 * @throws std::invalid_argument if T is not a supported tensor element type.
 */
template <typename T>
tensor::Dtype dtype_of() {
    if (std::is_same<T, float>::value)
        return tensor::Dtype::Float32;
    if (std::is_same<T, double>::value)
        return tensor::Dtype::Float64;
    if (std::is_same<T, int32_t>::value)
        return tensor::Dtype::Int32;
    if (std::is_same<T, int64_t>::value)
        return tensor::Dtype::Int64;
    throw std::invalid_argument("unsupported checkpoint element type");
}

/**
 * Write a little-endian integer or IEEE scalar.
 * Copies `sizeof(U)` bytes from `value` and reverses them on big-endian hosts.
 *
 * @param os Output stream.
 * @param value Value to write.
 *
 * @throws std::runtime_error if the stream fails.
 */
template <typename U>
void write_le(std::ostream& os, U value) {
    unsigned char bytes[sizeof(U)];
    std::memcpy(bytes, &value, sizeof(U));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    for (size_t i = 0; i < sizeof(U) / 2; ++i)
        std::swap(bytes[i], bytes[sizeof(U) - 1 - i]);
#endif
    os.write(reinterpret_cast<char*>(bytes), sizeof(U));
    if (!os)
        throw std::runtime_error("checkpoint: write failed");
}

/**
 * Read a little-endian integer or IEEE scalar.
 * Reverses bytes on big-endian hosts, then memcpy into `U`.
 *
 * @param is Input stream.
 * @return Decoded value.
 *
 * @throws std::runtime_error if the stream fails or hits EOF early.
 */
template <typename U>
U read_le(std::istream& is) {
    unsigned char bytes[sizeof(U)];
    is.read(reinterpret_cast<char*>(bytes), sizeof(U));
    if (!is)
        throw std::runtime_error("checkpoint: unexpected end of file");
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    for (size_t i = 0; i < sizeof(U) / 2; ++i)
        std::swap(bytes[i], bytes[sizeof(U) - 1 - i]);
#endif
    U value{};
    std::memcpy(&value, bytes, sizeof(U));
    return value;
}

/**
 * Write `n` bytes to `os`.
 *
 * @param os Output stream.
 * @param data Pointer to bytes; may be null when `n == 0`.
 * @param n Number of bytes.
 *
 * @throws std::runtime_error if the stream fails.
 */
inline void write_bytes(std::ostream& os, const char* data, std::streamsize n) {
    if (n == 0)
        return;
    os.write(data, n);
    if (!os)
        throw std::runtime_error("checkpoint: write failed");
}

/**
 * Read `n` bytes from `is` into `buf`.
 *
 * @param is Input stream.
 * @param buf Destination; may be null when `n == 0`.
 * @param n Number of bytes.
 *
 * @throws std::runtime_error if the stream fails or hits EOF early.
 */
inline void read_bytes(std::istream& is, char* buf, std::streamsize n) {
    if (n == 0)
        return;
    is.read(buf, n);
    if (!is)
        throw std::runtime_error("checkpoint: unexpected end of file");
}

} // namespace checkpoint_detail

/**
 * Snapshot every parameter and buffer on `layer` into packed records.
 * Walks `named_parameters()` then `named_buffers()`. Each record stores name,
 * shape, dtype, flags, and `to_vector()` values.
 *
 * @param layer Layer or module to snapshot.
 * @return Records in state_dict order.
 *
 * @throws std::invalid_argument if two tensors share the same dotted name.
 */
template <typename T>
std::vector<TensorRecord<T>> snapshot(Layer<T>& layer) {
    std::vector<TensorRecord<T>> records;
    std::unordered_set<std::string> seen;

    auto append = [&](NamedTensorList<T> named, bool is_buffer) {
        for (auto& kv : named) {
            if (!seen.insert(kv.first).second)
                throw std::invalid_argument(
                    "snapshot: duplicate state_dict key '" + kv.first + "'");
            tensor::Tensor<T>* t = kv.second;
            TensorRecord<T> rec;
            rec.name = kv.first;
            rec.shape = t->shape();
            rec.dtype = t->dtype();
            rec.requires_grad = t->requires_grad();
            rec.is_buffer = is_buffer;
            rec.values = t->to_vector();
            records.push_back(std::move(rec));
        }
    };

    append(layer.named_parameters(), false);
    append(layer.named_buffers(), true);
    return records;
}

/**
 * Write packed records to a binary checkpoint stream.
 *
 * @param os Binary output stream.
 * @param records Tensors to serialize.
 *
 * @throws std::runtime_error on I/O failure.
 * @throws std::invalid_argument if a record's values size does not match its shape.
 */
template <typename T>
void write_checkpoint(std::ostream& os, const std::vector<TensorRecord<T>>& records) {
    using checkpoint_detail::write_le;
    using checkpoint_detail::write_bytes;
    using checkpoint_detail::kMagic;
    using checkpoint_detail::kVersion;

    write_bytes(os, kMagic, 4);
    write_le<uint32_t>(os, kVersion);
    write_le<uint32_t>(os, static_cast<uint32_t>(records.size()));

    for (const auto& rec : records) {
        const int64_t expected = tensor::Tensor<T>::compute_numel(rec.shape);
        if (static_cast<int64_t>(rec.values.size()) != expected)
            throw std::invalid_argument(
                "write_checkpoint: '" + rec.name + "' values size does not match shape");
        if (rec.name.size() > checkpoint_detail::kMaxNameLen)
            throw std::invalid_argument(
                "write_checkpoint: '" + rec.name + "' name is too long");
        if (rec.shape.size() > checkpoint_detail::kMaxRank)
            throw std::invalid_argument(
                "write_checkpoint: '" + rec.name + "' rank is too large");

        write_le<uint32_t>(os, static_cast<uint32_t>(rec.name.size()));
        write_bytes(os, rec.name.data(), static_cast<std::streamsize>(rec.name.size()));
        write_le<uint32_t>(os, static_cast<uint32_t>(rec.dtype));
        write_le<uint8_t>(os, rec.requires_grad ? 1 : 0);
        write_le<uint8_t>(os, rec.is_buffer ? 1 : 0);
        write_le<uint16_t>(os, 0);
        write_le<uint32_t>(os, static_cast<uint32_t>(rec.shape.size()));
        for (int64_t dim : rec.shape)
            write_le<int64_t>(os, dim);
        write_le<int64_t>(os, expected);
        for (T v : rec.values)
            write_le<T>(os, v);
    }
}

/**
 * Read packed records from a binary checkpoint stream.
 * Validates magic, version, name length, rank, numel vs shape, and that every
 * tensor's dtype matches T.
 *
 * @param is Binary input stream.
 * @return Records in file order.
 *
 * @throws std::runtime_error on I/O failure or truncated file.
 * @throws std::invalid_argument on bad magic, version, layout, or dtype.
 */
template <typename T>
std::vector<TensorRecord<T>> read_checkpoint(std::istream& is) {
    using checkpoint_detail::read_le;
    using checkpoint_detail::read_bytes;
    using checkpoint_detail::kMagic;
    using checkpoint_detail::kVersion;
    using checkpoint_detail::dtype_of;
    using checkpoint_detail::dtype_name;

    char magic[4];
    read_bytes(is, magic, 4);
    if (std::memcmp(magic, kMagic, 4) != 0)
        throw std::invalid_argument("read_checkpoint: not an AGCK checkpoint");

    const uint32_t version = read_le<uint32_t>(is);
    if (version != kVersion)
        throw std::invalid_argument(
            "read_checkpoint: unsupported version " + std::to_string(version));

    const uint32_t n = read_le<uint32_t>(is);
    const tensor::Dtype expect_dtype = dtype_of<T>();
    std::vector<TensorRecord<T>> records;
    records.reserve(n);

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t name_len = read_le<uint32_t>(is);
        if (name_len > checkpoint_detail::kMaxNameLen)
            throw std::invalid_argument("read_checkpoint: name length is too large");
        std::string name(name_len, '\0');
        read_bytes(is, name.data(), static_cast<std::streamsize>(name_len));

        const auto dtype = static_cast<tensor::Dtype>(read_le<uint32_t>(is));
        const bool requires_grad = read_le<uint8_t>(is) != 0;
        const bool is_buffer = read_le<uint8_t>(is) != 0;
        (void)read_le<uint16_t>(is);

        const uint32_t rank = read_le<uint32_t>(is);
        if (rank > checkpoint_detail::kMaxRank)
            throw std::invalid_argument(
                "read_checkpoint: '" + name + "' rank is too large");
        std::vector<int64_t> shape(rank);
        for (uint32_t d = 0; d < rank; ++d)
            shape[d] = read_le<int64_t>(is);

        const int64_t numel = read_le<int64_t>(is);
        const int64_t expected = tensor::Tensor<T>::compute_numel(shape);
        if (numel != expected)
            throw std::invalid_argument(
                "read_checkpoint: '" + name + "' numel does not match shape");
        if (dtype != expect_dtype)
            throw std::invalid_argument(
                "read_checkpoint: '" + name + "' has dtype " + dtype_name(dtype) +
                ", expected " + dtype_name(expect_dtype));

        std::vector<T> values(static_cast<size_t>(numel));
        for (int64_t j = 0; j < numel; ++j)
            values[static_cast<size_t>(j)] = read_le<T>(is);

        TensorRecord<T> rec;
        rec.name = std::move(name);
        rec.shape = std::move(shape);
        rec.dtype = dtype;
        rec.requires_grad = requires_grad;
        rec.is_buffer = is_buffer;
        rec.values = std::move(values);
        records.push_back(std::move(rec));
    }
    return records;
}

/**
 * Copy checkpoint records into an already-constructed `layer`.
 * Matches by dotted name. Strict mode requires identical key sets and that
 * shape, dtype, `requires_grad`, and parameter-vs-buffer kind agree; then
 * `copy_` writes values without replacing tensor objects.
 *
 * @param layer Destination layer or module (must already have the right architecture).
 * @param records Checkpoint tensors.
 * @param strict If true (default), missing/unexpected keys and metadata
 *               mismatches throw. If false, matching keys are copied and the
 *               rest are skipped (warnings when logging is enabled).
 *
 * @throws std::invalid_argument on duplicate checkpoint names.
 * @throws std::invalid_argument in strict mode on key or structure mismatch.
 * @throws std::invalid_argument if a matching tensor's shape or dtype differs.
 */
template <typename T>
void load_state_dict(
    Layer<T>& layer,
    const std::vector<TensorRecord<T>>& records,
    bool strict = true
) {
    using checkpoint_detail::format_shape;
    using checkpoint_detail::dtype_name;

    std::unordered_map<std::string, tensor::Tensor<T>*> dest;
    std::unordered_map<std::string, bool> dest_is_buffer;
    dest.reserve(records.size());

    auto index = [&](const NamedTensorList<T>& named, bool is_buffer) {
        for (auto& kv : named) {
            if (dest.count(kv.first))
                throw std::invalid_argument(
                    "load_state_dict: duplicate model key '" + kv.first + "'");
            dest[kv.first] = kv.second;
            dest_is_buffer[kv.first] = is_buffer;
        }
    };
    index(layer.named_parameters(), false);
    index(layer.named_buffers(), true);

    std::unordered_set<std::string> src_names;
    std::vector<std::string> errors;
    std::vector<std::pair<tensor::Tensor<T>*, const TensorRecord<T>*>> pending;

    for (const auto& rec : records) {
        if (!src_names.insert(rec.name).second)
            throw std::invalid_argument(
                "load_state_dict: duplicate checkpoint key '" + rec.name + "'");
        auto it = dest.find(rec.name);
        if (it == dest.end()) {
            if (strict)
                errors.push_back("unexpected key '" + rec.name + "'");
            else
                logging::warning("load_state_dict: skipping unexpected key '" + rec.name + "'");
            continue;
        }
        tensor::Tensor<T>* t = it->second;
        const bool model_is_buffer = dest_is_buffer[rec.name];
        if (t->shape() != rec.shape)
            errors.push_back(
                "size mismatch for '" + rec.name + "': checkpoint " +
                format_shape(rec.shape) + ", model " + format_shape(t->shape()));
        else if (t->dtype() != rec.dtype)
            errors.push_back(
                "dtype mismatch for '" + rec.name + "': checkpoint " +
                dtype_name(rec.dtype) + ", model " + dtype_name(t->dtype()));
        else if (t->requires_grad() != rec.requires_grad)
            errors.push_back(
                "requires_grad mismatch for '" + rec.name + "'");
        else if (model_is_buffer != rec.is_buffer)
            errors.push_back(
                std::string(rec.is_buffer ? "buffer" : "parameter") +
                " '" + rec.name + "' does not match the model");
        else
            pending.emplace_back(t, &rec);
    }

    for (const auto& kv : dest) {
        if (src_names.count(kv.first))
            continue;
        if (strict)
            errors.push_back("missing key '" + kv.first + "'");
        else
            logging::warning("load_state_dict: missing key '" + kv.first + "'");
    }

    if (!errors.empty()) {
        std::ostringstream oss;
        oss << "Error(s) in loading state_dict:\n";
        for (const auto& e : errors)
            oss << "    " << e << "\n";
        throw std::invalid_argument(oss.str());
    }

    for (auto& item : pending)
        item.first->copy_(tensor::Tensor<T>(
            item.second->shape, item.second->values, item.second->requires_grad));
}

/**
 * Write `layer`'s state_dict to `path` as an AGCK checkpoint.
 * Snapshots parameters and buffers, then writes packed values plus structure.
 *
 * @param layer Layer or module to save.
 * @param path Destination file path.
 *
 * @throws std::runtime_error if the file cannot be created or written.
 * @throws std::invalid_argument if the model has duplicate state_dict keys.
 */
template <typename T>
void save(Layer<T>& layer, const std::string& path) {
    auto records = snapshot(layer);
    std::ofstream os(path, std::ios::binary);
    if (!os)
        throw std::runtime_error("save: cannot open '" + path + "' for writing");
    write_checkpoint(os, records);
    os.close();
    if (!os)
        throw std::runtime_error("save: failed writing '" + path + "'");
    logging::info(
        "saved " + std::to_string(records.size()) + " tensors (" +
        std::to_string(layer.parameters().size()) + " parameters) to " + path);
}

/**
 * Load an AGCK checkpoint from `path` into `layer`.
 * Reconstructs records, then `load_state_dict` copies values after checking
 * that names and tensor structure match the live model.
 *
 * @param layer Destination layer or module (architecture must already match).
 * @param path Checkpoint file path.
 * @param strict If true (default), require an exact key/structure match.
 *
 * @throws std::runtime_error if the file cannot be opened.
 * @throws std::invalid_argument on format errors or structure mismatch.
 */
template <typename T>
void load(Layer<T>& layer, const std::string& path, bool strict = true) {
    std::ifstream is(path, std::ios::binary);
    if (!is)
        throw std::runtime_error("load: cannot open '" + path + "' for reading");
    auto records = read_checkpoint<T>(is);
    load_state_dict(layer, records, strict);
    logging::info(
        "loaded " + std::to_string(records.size()) + " tensors from " + path);
}

} // namespace nn
