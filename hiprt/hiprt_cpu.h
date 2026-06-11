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

#include <hiprt/hiprt_types.h>
#include <hiprt/impl/CpuTypes.h>
#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>

// ---------------------------------------------------------------------------
// Internal helpers – not for external consumption
// ---------------------------------------------------------------------------
namespace hiprt_cpu_detail
{

inline RTCRayHit makeRTCRayHit( const hiprtRay& ray ) noexcept
{
	RTCRayHit rh{};
	rh.ray.org_x	 = ray.origin.x;
	rh.ray.org_y	 = ray.origin.y;
	rh.ray.org_z	 = ray.origin.z;
	rh.ray.dir_x	 = ray.direction.x;
	rh.ray.dir_y	 = ray.direction.y;
	rh.ray.dir_z	 = ray.direction.z;
	rh.ray.tnear	 = ray.minT;
	rh.ray.tfar		 = ray.maxT;
	rh.ray.mask		 = static_cast<unsigned>( -1 );
	rh.ray.flags	 = 0;
	rh.hit.geomID	 = RTC_INVALID_GEOMETRY_ID;
	rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
	return rh;
}

inline hiprtHit makeHiprtHit( const RTCRayHit& rh ) noexcept
{
	hiprtHit hit{};
	if ( rh.hit.geomID == RTC_INVALID_GEOMETRY_ID )
		return hit; // primID stays hiprtInvalidValue -> hasHit() == false

	hit.primID		= rh.hit.primID;
	hit.instanceID	= ( rh.hit.instID[0] != RTC_INVALID_GEOMETRY_ID )
						  ? rh.hit.instID[0]
						  : hiprtInvalidValue;
	hit.uv.x		= rh.hit.u;
	hit.uv.y		= rh.hit.v;
	hit.normal.x	= rh.hit.Ng_x;
	hit.normal.y	= rh.hit.Ng_y;
	hit.normal.z	= rh.hit.Ng_z;
	hit.t			= rh.ray.tfar; // Embree writes the hit distance into tfar
	return hit;
}

} // namespace hiprt_cpu_detail

// ---------------------------------------------------------------------------
// Public CPU traversal objects
// ---------------------------------------------------------------------------

/** \brief Host-side traversal that finds the closest hit in a hiprtGeometry.
 *
 * Mirrors the GPU class hiprtGeomTraversalClosest. Constructing the object
 * immediately casts the handle to CpuGeometryData, retrieves the underlying
 * RTCScene built by CpuContext, and calls rtcIntersect1.
 * Call getNextHit() to obtain the result.
 */
class hiprtGeomTraversalClosestCPU
{
  public:
	hiprtGeomTraversalClosestCPU( hiprtGeometry geom, const hiprtRay& ray ) noexcept
	{
		m_rayHit = hiprt_cpu_detail::makeRTCRayHit( ray );
		auto* data = reinterpret_cast<hiprt::CpuGeometryData*>( geom );
		if ( data != nullptr && data->rtcScene != nullptr )
			rtcIntersect1( data->rtcScene, &m_rayHit );
	}

	hiprtHit getNextHit() const noexcept
	{
		return hiprt_cpu_detail::makeHiprtHit( m_rayHit );
	}

  private:
	RTCRayHit m_rayHit{};
};

/** \brief Host-side traversal that finds the closest hit in a hiprtScene.
 *
 * Mirrors the GPU class hiprtSceneTraversalClosest. Constructing the object
 * immediately casts the handle to CpuSceneData, retrieves the underlying
 * RTCScene built by CpuContext (which already encodes the full instance
 * hierarchy via RTC_GEOMETRY_TYPE_INSTANCE), and calls rtcIntersect1.
 * Call getNextHit() to obtain the result, including instanceID populated
 * from instID[0].
 */
class hiprtSceneTraversalClosestCPU
{
  public:
	hiprtSceneTraversalClosestCPU( hiprtScene scene, const hiprtRay& ray ) noexcept
	{
		m_rayHit = hiprt_cpu_detail::makeRTCRayHit( ray );
		auto* data = reinterpret_cast<hiprt::CpuSceneData*>( scene );
		if ( data != nullptr && data->rtcScene != nullptr )
			rtcIntersect1( data->rtcScene, &m_rayHit );
	}

	hiprtHit getNextHit() const noexcept
	{
		return hiprt_cpu_detail::makeHiprtHit( m_rayHit );
	}

  private:
	RTCRayHit m_rayHit{};
};
