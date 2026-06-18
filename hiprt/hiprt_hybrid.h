//////////////////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.
//
//////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hiprt/hiprt.h>

/** \brief Scheduling parameters for hybrid closest-hit tracing.
 */
struct hiprtHybridTraceConfig
{
	/*!< Fraction of rays dispatched to the GPU in [0, 1]. */
	float gpuFraction = 0.8f;
	/*!< Minimum number of rays kept for the CPU path when both backends are used. */
	uint32_t minCpuBatch = 1024u;
};

/** \brief GPU resources required for hybrid scene tracing.
 *
 * Compile hiprt/kernels/HybridSceneTraceBatchKernel.h with hiprtBuildTraceKernels
 * and pass the resulting function handle here. globalStack is unused by the
 * default batch kernel but reserved for advanced kernels such as TraceKernel.
 */
struct hiprtHybridTraceGpuInput
{
	hiprtApiFunction	   traceKernel	 = nullptr;
	hiprtGlobalStackBuffer globalStack = {};
};

/** \brief Trace a batch of closest-hit scene rays on GPU and CPU concurrently.
 *
 * Requires a hybrid context (hiprtDeviceAMD | hiprtDeviceCPU or
 * hiprtDeviceNVIDIA | hiprtDeviceCPU). The first gpuCount rays are traced on
 * the GPU; the remainder on the CPU Embree mirror. rays and hits must be
 * host-visible (managed/UVA or pinned mapped) so both backends can access them.
 *
 * \param context Hybrid HIPRT context.
 * \param scene GPU scene handle with a CPU mirror.
 * \param config Split policy between GPU and CPU.
 * \param gpuInput Precompiled GPU trace kernel (and optional stack buffer).
 * \param rays Host-visible ray array of length rayCount.
 * \param hits Host-visible hit array of length rayCount.
 * \param rayCount Number of rays to trace.
 * \param stream GPU stream for the device launch (0 for default stream).
 * \return hiprtSuccess or an error code.
 */
HIPRT_API hiprtError hiprtTraceHybridClosest(
	hiprtContext					 context,
	hiprtScene						 scene,
	const hiprtHybridTraceConfig&	 config,
	const hiprtHybridTraceGpuInput&	 gpuInput,
	hiprtRay*						 rays,
	hiprtHit*						 hits,
	uint32_t						 rayCount,
	hiprtApiStream					 stream );
