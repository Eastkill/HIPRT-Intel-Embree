#pragma once

#include <hiprt/hiprt_types.h>

namespace hiprt
{
struct CpuGeometryData;
struct CpuSceneData;

/**
 * Global CPU-data registry.
 *
 * GPU pointers (oroDeviceptr cast to hiprtGeometry/hiprtScene) are unique
 * process-wide, so a single flat map is sufficient. Pure-CPU handles are the
 * CpuGeometryData/CpuSceneData pointers themselves — they are also unique.
 * Both can therefore share the same table without ambiguity.
 *
 * All functions are thread-safe via an internal mutex.
 */

void registerCpuGeom( hiprtGeometry handle, CpuGeometryData* data );
void unregisterCpuGeom( hiprtGeometry handle );
CpuGeometryData* lookupCpuGeom( hiprtGeometry handle );

void registerCpuScene( hiprtScene handle, CpuSceneData* data );
void unregisterCpuScene( hiprtScene handle );
CpuSceneData* lookupCpuScene( hiprtScene handle );

} // namespace hiprt
