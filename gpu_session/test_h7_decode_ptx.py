#!/usr/bin/env python3
"""test_h7_decode_ptx.py -- H7: can decode.ptx be loaded and run WITHOUT
CuPy, through the raw CUDA driver API from Python via ctypes (standing in
for what a Rust binary would do via FFI to libcuda.so -- same C ABI, no
compiler needed on the rented box)?

Method (pre-registered in gpu_session/EXPECTED.md, read that first):
  1. Load one real DF11 unit's five kernel-input tensors (luts,
     encoded_exponent, sign_mantissa, output_positions, gaps) straight off
     disk as raw bytes -- no torch, no reinterpretation.
  2. CONTROL: decode them with real CuPy (RawModule + RawKernel), exactly
     the call dfloat11.py itself makes.
  3. QUESTION: decode the same bytes with the CUDA driver API reached
     directly via ctypes against libcuda.so -- cuInit, cuCtxCreate,
     cuModuleLoadData(decode.ptx), cuModuleGetFunction, cuMemAlloc/
     cuMemcpyHtoD, cuLaunchKernel, cuMemcpyDtoH. No CuPy, no torch, no
     compiler.
  4. Compare the two raw output byte buffers bit-for-bit.

Kernel signature (read directly from decode.ptx's own .visible .entry
declaration, not guessed):
    decode(u64 luts, u64 encoded_exponent, u64 sign_mantissa,
           u64 output_positions, u64 gaps, u64 out,
           u32 n_luts, u32 n_bytes, u32 n_elements)
launched with block=(512,1,1), grid=(ceil(n_bytes/(512*8)),1,1), and
shared_mem_size = 512*4 + 4 + 2*max(output_positions deltas as uint32) --
all taken verbatim from dfloat11.py's own launch code (get_hook and
compress_model's check_correctness branch), not re-derived.

Usage:
    test_h7_decode_ptx.py [--unit-file PATH] [--out PATH]

Default --unit-file is the uploaded tier0 shard
phase0/out/official/qwen3-trunc-layers-only-dir/model_layers_0.safetensors
resolved relative to the repo root two levels up from this file.
"""
import argparse
import ctypes
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_unit_file():
    return os.path.join(
        repo_root(), "phase0", "out", "official",
        "qwen3-trunc-layers-only-dir", "model_layers_0.safetensors",
    )


# ---------------------------------------------------------------------------
# CONTROL: CuPy path -- shared with test_luts_correct_kernel.py via common.py
# ---------------------------------------------------------------------------
decode_with_cupy = common.decode_with_cupy


# ---------------------------------------------------------------------------
# QUESTION: raw CUDA driver API via ctypes -- no CuPy, no torch
# ---------------------------------------------------------------------------
CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64
CUstream = ctypes.c_void_p


def load_libcuda():
    last_err = None
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(name)
        except OSError as e:
            last_err = e
    raise RuntimeError(
        f"libcuda.so(.1) not found -- is the NVIDIA driver installed on this box? ({last_err})"
    )


def setup_prototypes(cuda):
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = CUresult

    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(CUdevice), ctypes.c_int]
    cuda.cuDeviceGet.restype = CUresult

    ctx_create = getattr(cuda, "cuCtxCreate_v2", None) or cuda.cuCtxCreate
    ctx_create.argtypes = [ctypes.POINTER(CUcontext), ctypes.c_uint, CUdevice]
    ctx_create.restype = CUresult
    cuda._ctx_create = ctx_create

    cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(CUmodule), ctypes.c_char_p]
    cuda.cuModuleLoadData.restype = CUresult

    cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(CUfunction), CUmodule, ctypes.c_char_p]
    cuda.cuModuleGetFunction.restype = CUresult

    mem_alloc = getattr(cuda, "cuMemAlloc_v2", None) or cuda.cuMemAlloc
    mem_alloc.argtypes = [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t]
    mem_alloc.restype = CUresult
    cuda._mem_alloc = mem_alloc

    h2d = getattr(cuda, "cuMemcpyHtoD_v2", None) or cuda.cuMemcpyHtoD
    h2d.argtypes = [CUdeviceptr, ctypes.c_void_p, ctypes.c_size_t]
    h2d.restype = CUresult
    cuda._h2d = h2d

    d2h = getattr(cuda, "cuMemcpyDtoH_v2", None) or cuda.cuMemcpyDtoH
    d2h.argtypes = [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t]
    d2h.restype = CUresult
    cuda._d2h = d2h

    mem_free = getattr(cuda, "cuMemFree_v2", None) or cuda.cuMemFree
    mem_free.argtypes = [CUdeviceptr]
    mem_free.restype = CUresult
    cuda._mem_free = mem_free

    cuda.cuLaunchKernel.argtypes = [
        CUfunction,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, CUstream,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ]
    cuda.cuLaunchKernel.restype = CUresult

    cuda.cuCtxSynchronize.argtypes = []
    cuda.cuCtxSynchronize.restype = CUresult

    cuda.cuGetErrorString.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
    cuda.cuGetErrorString.restype = CUresult


def cu_check(cuda, code, what):
    if code != 0:
        msg = ctypes.c_char_p()
        try:
            cuda.cuGetErrorString(code, ctypes.byref(msg))
            m = msg.value.decode() if msg.value else "?"
        except Exception:
            m = "?"
        raise RuntimeError(f"driver API call failed: {what} -> CUresult {code} ({m})")


def decode_with_driver_api(unit_bytes, n_elements):
    cuda = load_libcuda()
    setup_prototypes(cuda)

    cu_check(cuda, cuda.cuInit(0), "cuInit")
    dev = CUdevice()
    cu_check(cuda, cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = CUcontext()
    cu_check(cuda, cuda._ctx_create(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

    ptx_path = common.find_ptx_path()
    with open(ptx_path, "rb") as f:
        ptx_bytes = f.read()
    if not ptx_bytes.endswith(b"\x00"):
        ptx_bytes += b"\x00"

    module = CUmodule()
    cu_check(cuda, cuda.cuModuleLoadData(ctypes.byref(module), ptx_bytes), "cuModuleLoadData")
    func = CUfunction()
    cu_check(cuda, cuda.cuModuleGetFunction(ctypes.byref(func), module, b"decode"), "cuModuleGetFunction")

    n_luts, n_bytes, grid, block, shared_mem = common.launch_geometry(
        unit_bytes["luts"], unit_bytes["encoded_exponent"], unit_bytes["output_positions"],
    )
    out_nbytes = n_elements * 2

    device_ptrs = {}
    order = ("luts", "encoded_exponent", "sign_mantissa", "output_positions", "gaps")
    try:
        for name in order:
            data = unit_bytes[name]
            dptr = CUdeviceptr()
            cu_check(cuda, cuda._mem_alloc(ctypes.byref(dptr), len(data)), f"cuMemAlloc({name})")
            device_ptrs[name] = dptr
            buf = ctypes.create_string_buffer(data, len(data))
            buf_vp = ctypes.cast(buf, ctypes.c_void_p)
            cu_check(cuda, cuda._h2d(dptr, buf_vp, len(data)), f"cuMemcpyHtoD({name})")

        out_dptr = CUdeviceptr()
        cu_check(cuda, cuda._mem_alloc(ctypes.byref(out_dptr), out_nbytes), "cuMemAlloc(out)")

        # kernel params, in decode.ptx's declared order:
        # 6x u64 pointer, then n_luts, n_bytes, n_elements as u32
        ptr_params = [device_ptrs[n] for n in order] + [out_dptr]
        scalar_params = [
            ctypes.c_uint32(n_luts),
            ctypes.c_uint32(n_bytes),
            ctypes.c_uint32(n_elements),
        ]
        all_params = list(ptr_params) + scalar_params
        kernel_params = (ctypes.c_void_p * len(all_params))(
            *[ctypes.cast(ctypes.pointer(p), ctypes.c_void_p) for p in all_params]
        )

        cu_check(cuda, cuda.cuLaunchKernel(
            func,
            grid[0], 1, 1,
            block[0], 1, 1,
            shared_mem, None,
            kernel_params, None,
        ), "cuLaunchKernel")
        cu_check(cuda, cuda.cuCtxSynchronize(), "cuCtxSynchronize")

        out_buf = ctypes.create_string_buffer(out_nbytes)
        out_buf_vp = ctypes.cast(out_buf, ctypes.c_void_p)
        cu_check(cuda, cuda._d2h(out_buf_vp, out_dptr, out_nbytes), "cuMemcpyDtoH(out)")
        return out_buf.raw
    finally:
        for dptr in list(device_ptrs.values()):
            try:
                cuda._mem_free(dptr)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-file", default=default_unit_file())
    ap.add_argument("--out", default=common.default_out_path("h7_decode_ptx"))
    args = ap.parse_args()

    detail = {"unit_file": args.unit_file, "gpu": common.gpu_info()}

    with common.Timer() as t:
        try:
            if not os.path.exists(args.unit_file):
                raise FileNotFoundError(
                    f"unit file not found: {args.unit_file} "
                    "(did you upload phase0/out/official/qwen3-trunc-layers-only-dir/ ?)"
                )
            header, data_start = common.read_header(args.unit_file)
            prefix = common.only_unit_prefix(header)
            unit_bytes, tensor_infos = common.load_unit_raw(
                args.unit_file, prefix, header=header, data_start=data_start,
            )
            n_elements = len(unit_bytes["sign_mantissa"])
            detail["unit_prefix"] = prefix
            detail["n_elements"] = n_elements
            detail["n_bytes_encoded_exponent"] = len(unit_bytes["encoded_exponent"])
            detail["n_luts"] = len(unit_bytes["luts"]) // 256

            print(f"[H7] decoding unit {prefix!r} ({n_elements} elements) via CuPy (control)...")
            cupy_out = decode_with_cupy(unit_bytes, n_elements)
            detail["cupy_output_sha256"] = common.sha256_hex(cupy_out)
            detail["cupy_output_len"] = len(cupy_out)

            print(f"[H7] decoding the same unit via raw CUDA driver API / ctypes (question)...")
            driver_out = decode_with_driver_api(unit_bytes, n_elements)
            detail["driver_api_output_sha256"] = common.sha256_hex(driver_out)
            detail["driver_api_output_len"] = len(driver_out)

            identical = common.bytes_equal(cupy_out, driver_out)
            detail["bit_for_bit_identical"] = identical

            if identical:
                status = "CONFIRMED"
                summary = (
                    f"decode.ptx loaded and executed via raw CUDA driver API (ctypes, no CuPy) "
                    f"produced output bit-for-bit identical to CuPy's decode of the same unit "
                    f"({n_elements} elements, sha256 {detail['cupy_output_sha256'][:16]}...)."
                )
            else:
                status = "REFUTED"
                summary = (
                    "decode.ptx ran via the driver API without CuPy, but its output DIFFERED "
                    "from CuPy's decode of the identical input -- the two paths are not "
                    "equivalent."
                )
        except Exception as e:
            status = "ERROR"
            summary = f"{e.__class__.__name__}: {e}"
            detail["traceback"] = traceback.format_exc()

    common.write_result(args.out, "H7_decode_ptx_without_cupy", status, summary, detail, t.elapsed)
    sys.exit(0 if status in ("CONFIRMED", "REFUTED") else 1)


if __name__ == "__main__":
    main()
