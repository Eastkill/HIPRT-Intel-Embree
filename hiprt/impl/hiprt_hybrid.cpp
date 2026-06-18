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

#include <hiprt/hiprt_hybrid.h>
#include <hiprt/hiprt_cpu.h>
#include <hiprt/impl/HybridContext.h>

#include <Orochi/Orochi.h>

using namespace hiprt;

namespace
{
ContextBase* asContext( hiprtContext context ) noexcept
{
	return reinterpret_cast<ContextBase*>( context );
}

uint32_t computeGpuRayCount( uint32_t rayCount, const hiprtHybridTraceConfig& config ) noexcept
{
	if ( rayCount == 0 )
		return 0;

	uint32_t gpuCount = static_cast<uint32_t>( static_cast<float>( rayCount ) * config.gpuFraction );
	if ( gpuCount > rayCount )
		gpuCount = rayCount;

	const uint32_t cpuCount = rayCount - gpuCount;
	if ( cpuCount > 0 && cpuCount < config.minCpuBatch && rayCount > config.minCpuBatch )
	{
		gpuCount = rayCount - config.minCpuBatch;
		if ( gpuCount > rayCount )
			gpuCount = rayCount;
	}

	return gpuCount;
}

hiprtError launchSceneTraceBatchKernel(
	const hiprtHybridTraceGpuInput& gpuInput,
	hiprtScene						scene,
	uint32_t						numRays,
	hiprtRay*						rays,
	hiprtHit*						hits,
	hiprtApiStream					stream )
{
	if ( gpuInput.traceKernel == nullptr )
		return hiprtErrorInvalidParameter;

	constexpr uint32_t blockSize = 256u;
	const uint32_t	   gridSize  = ( numRays + blockSize - 1 ) / blockSize;

	hiprtApiFunction funcApi = gpuInput.traceKernel;
	oroFunction		 func	= *reinterpret_cast<oroFunction*>( &funcApi );

	void* args[] = { &scene, &numRays, &rays, &hits };
	const oroError err = oroModuleLaunchKernel(
		func, gridSize, 1, 1, blockSize, 1, 1, 0, reinterpret_cast<oroStream>( stream ), args, 0 );
	return ( err == oroSuccess ) ? hiprtSuccess : hiprtErrorInternal;
}
} // namespace

hiprtError hiprtTraceHybridClosest(
	hiprtContext					 context,
	hiprtScene						 scene,
	const hiprtHybridTraceConfig&	 config,
	const hiprtHybridTraceGpuInput&	 gpuInput,
	hiprtRay*						 rays,
	hiprtHit*						 hits,
	uint32_t						 rayCount,
	hiprtApiStream					 stream )
{
	if ( context == nullptr || scene == nullptr || rays == nullptr || hits == nullptr )
		return hiprtErrorInvalidParameter;

	if ( dynamic_cast<HybridContext*>( asContext( context ) ) == nullptr )
		return hiprtErrorInvalidParameter;

	if ( hiprtGetInternalCpuDataFromScene( scene ) == nullptr )
		return hiprtErrorInvalidParameter;

	const uint32_t gpuCount = computeGpuRayCount( rayCount, config );
	const uint32_t cpuCount = rayCount - gpuCount;

	if ( gpuCount > 0 )
	{
		const hiprtError gpuErr =
			launchSceneTraceBatchKernel( gpuInput, scene, gpuCount, rays, hits, stream );
		if ( gpuErr != hiprtSuccess )
			return gpuErr;

		const oroStream gpuStream = reinterpret_cast<oroStream>( stream );
		const oroError	err		  = ( gpuStream != nullptr ) ? oroStreamSynchronize( gpuStream ) : oroDeviceSynchronize();
		if ( err != oroSuccess )
			return hiprtErrorInternal;
	}

	if ( cpuCount > 0 )
	{
		hiprtGeomTraversalClosestCPU::traceBatch(
			context, scene, rays + gpuCount, hits + gpuCount, cpuCount );
	}

	return hiprtSuccess;
}
