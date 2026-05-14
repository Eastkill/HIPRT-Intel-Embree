#include "CpuContext.h"

#include <stdexcept>

namespace hiprt
{
namespace
{
[[noreturn]] void throwNotImplemented( const char* what )
{
	throw std::runtime_error( what );
}
} // namespace

std::vector<hiprtGeometry> CpuContext::createGeometries(
	const std::vector<hiprtGeometryBuildInput>&, const hiprtBuildOptions )
{
	throwNotImplemented( "CpuContext::createGeometries is not implemented yet." );
}

void CpuContext::destroyGeometries( const std::vector<hiprtGeometry>& ) { throwNotImplemented( "CpuContext::destroyGeometries is not implemented yet." ); }

void CpuContext::buildGeometries(
	const std::vector<hiprtGeometryBuildInput>&,
	const hiprtBuildOptions,
	hiprtDevicePtr,
	oroStream,
	std::vector<hiprtDevicePtr>& )
{
	throwNotImplemented( "CpuContext::buildGeometries is not implemented yet." );
}

void CpuContext::updateGeometries(
	const std::vector<hiprtGeometryBuildInput>&,
	const hiprtBuildOptions,
	hiprtDevicePtr,
	oroStream,
	std::vector<hiprtDevicePtr>& )
{
	throwNotImplemented( "CpuContext::updateGeometries is not implemented yet." );
}

size_t CpuContext::getGeometriesBuildTempBufferSize( const std::vector<hiprtGeometryBuildInput>&, const hiprtBuildOptions )
{
	throwNotImplemented( "CpuContext::getGeometriesBuildTempBufferSize is not implemented yet." );
}

std::vector<hiprtGeometry> CpuContext::compactGeometries( const std::vector<hiprtGeometry>&, oroStream )
{
	throwNotImplemented( "CpuContext::compactGeometries is not implemented yet." );
}

std::vector<hiprtScene> CpuContext::createScenes( const std::vector<hiprtSceneBuildInput>&, const hiprtBuildOptions )
{
	throwNotImplemented( "CpuContext::createScenes is not implemented yet." );
}

void CpuContext::destroyScenes( const std::vector<hiprtScene>& ) { throwNotImplemented( "CpuContext::destroyScenes is not implemented yet." ); }

void CpuContext::buildScenes(
	const std::vector<hiprtSceneBuildInput>&,
	const hiprtBuildOptions,
	hiprtDevicePtr,
	oroStream,
	std::vector<hiprtDevicePtr>& )
{
	throwNotImplemented( "CpuContext::buildScenes is not implemented yet." );
}

void CpuContext::updateScenes(
	const std::vector<hiprtSceneBuildInput>&,
	const hiprtBuildOptions,
	hiprtDevicePtr,
	oroStream,
	std::vector<hiprtDevicePtr>& )
{
	throwNotImplemented( "CpuContext::updateScenes is not implemented yet." );
}

size_t CpuContext::getScenesBuildTempBufferSize( const std::vector<hiprtSceneBuildInput>&, const hiprtBuildOptions )
{
	throwNotImplemented( "CpuContext::getScenesBuildTempBufferSize is not implemented yet." );
}

std::vector<hiprtScene> CpuContext::compactScenes( const std::vector<hiprtScene>&, oroStream )
{
	throwNotImplemented( "CpuContext::compactScenes is not implemented yet." );
}

hiprtFuncTable CpuContext::createFuncTable( uint32_t, uint32_t )
{
	throwNotImplemented( "CpuContext::createFuncTable is not implemented yet." );
}

void CpuContext::setFuncTable( hiprtFuncTable, uint32_t, uint32_t, hiprtFuncDataSet )
{
	throwNotImplemented( "CpuContext::setFuncTable is not implemented yet." );
}

void CpuContext::destroyFuncTable( hiprtFuncTable ) { throwNotImplemented( "CpuContext::destroyFuncTable is not implemented yet." ); }

void CpuContext::createGlobalStackBuffer( const hiprtGlobalStackBufferInput&, hiprtGlobalStackBuffer& )
{
	throwNotImplemented( "CpuContext::createGlobalStackBuffer is not implemented yet." );
}

void CpuContext::destroyGlobalStackBuffer( hiprtGlobalStackBuffer ) { throwNotImplemented( "CpuContext::destroyGlobalStackBuffer is not implemented yet." ); }

void CpuContext::saveGeometry( hiprtGeometry, const std::string& ) { throwNotImplemented( "CpuContext::saveGeometry is not implemented yet." ); }

hiprtGeometry CpuContext::loadGeometry( const std::string& ) { throwNotImplemented( "CpuContext::loadGeometry is not implemented yet." ); }

void CpuContext::saveScene( hiprtScene, const std::string& ) { throwNotImplemented( "CpuContext::saveScene is not implemented yet." ); }

hiprtScene CpuContext::loadScene( const std::string& ) { throwNotImplemented( "CpuContext::loadScene is not implemented yet." ); }

void CpuContext::exportGeometryAabb( hiprtGeometry, float3&, float3& )
{
	throwNotImplemented( "CpuContext::exportGeometryAabb is not implemented yet." );
}

void CpuContext::exportSceneAabb( hiprtScene, float3&, float3& ) { throwNotImplemented( "CpuContext::exportSceneAabb is not implemented yet." ); }

void CpuContext::buildKernels(
	const std::vector<const char*>&,
	const std::string&,
	const std::filesystem::path&,
	std::vector<const char*>&,
	std::vector<const char*>&,
	std::vector<const char*>&,
	uint32_t,
	uint32_t,
	const std::vector<hiprtFuncNameSet>&,
	std::vector<oroFunction>&,
	oroModule&,
	bool )
{
	throwNotImplemented( "CpuContext::buildKernels is not implemented yet." );
}

void CpuContext::buildKernelsFromBitcode(
	const std::vector<const char*>&,
	const std::filesystem::path&,
	const std::string_view,
	uint32_t,
	uint32_t,
	const std::vector<hiprtFuncNameSet>&,
	std::vector<oroFunction>&,
	bool )
{
	throwNotImplemented( "CpuContext::buildKernelsFromBitcode is not implemented yet." );
}

} // namespace hiprt
