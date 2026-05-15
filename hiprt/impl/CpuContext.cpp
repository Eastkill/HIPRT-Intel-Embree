#include "CpuContext.h"

#include <iostream>
#include <stdexcept>

namespace hiprt
{
namespace
{
[[noreturn]] void throwNotImplemented( const char* what )
{
	std::cerr << "[CpuContext] not implemented: " << what << std::endl;
	throw std::runtime_error( what );
}
} // namespace

CpuContext::CpuContext( const hiprtContextCreationInput& input )
{
	std::string config;
	if ( input.numCpuThreads != 0 )
		config = "threads=" + std::to_string( input.numCpuThreads );

	m_rtcDevice = rtcNewDevice( config.empty() ? nullptr : config.c_str() );

	if ( m_rtcDevice == nullptr )
	{
		const RTCError err = rtcGetDeviceError( nullptr );
		std::cerr << "[CpuContext] rtcNewDevice failed, RTCError=" << static_cast<int>( err ) << std::endl;
		throw std::runtime_error( "rtcNewDevice failed" );
	}

	std::cerr << "[CpuContext] constructed (numCpuThreads=" << input.numCpuThreads
			  << ( input.numCpuThreads == 0 ? " -> all available)" : ")" ) << std::endl;
}

CpuContext::~CpuContext()
{
	if ( m_rtcDevice != nullptr )
	{
		rtcReleaseDevice( m_rtcDevice );
		m_rtcDevice = nullptr;
	}
}

std::vector<hiprtGeometry> CpuContext::createGeometries(
	const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions )
{
	std::vector<hiprtGeometry> geometries;
	geometries.reserve( buildInputs.size() );

	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		CpuGeometryData* data = new CpuGeometryData();
		data->rtcDevice		  = m_rtcDevice;
		data->rtcScene		  = rtcNewScene( m_rtcDevice );

		if ( data->rtcScene == nullptr )
		{
			const RTCError err = rtcGetDeviceError( m_rtcDevice );
			delete data;
			std::cerr << "[CpuContext] rtcNewScene failed at index " << i
					  << ", RTCError=" << static_cast<int>( err ) << std::endl;
			throw std::runtime_error( "rtcNewScene failed" );
		}

		geometries.push_back( reinterpret_cast<hiprtGeometry>( data ) );
	}

	return geometries;
}

void CpuContext::destroyGeometries( const std::vector<hiprtGeometry>& geometries )
{
	for ( hiprtGeometry geometry : geometries )
	{
		CpuGeometryData* data = reinterpret_cast<CpuGeometryData*>( geometry );
		if ( data == nullptr ) continue;

		if ( data->rtcScene != nullptr )
		{
			rtcReleaseScene( data->rtcScene );
			data->rtcScene = nullptr;
		}

		delete data;
	}
}

void CpuContext::buildGeometryEntry( CpuGeometryData* data, const hiprtGeometryBuildInput& input, size_t index )
{
	if ( data == nullptr || data->rtcScene == nullptr )
		throw std::runtime_error( "CpuContext::buildGeometryEntry: invalid geometry handle" );

	if ( input.type == hiprtPrimitiveTypeTriangleMesh )
	{
		const hiprtTriangleMeshPrimitive& mesh = input.primitive.triangleMesh;

		RTCGeometry geom = rtcNewGeometry( m_rtcDevice, RTC_GEOMETRY_TYPE_TRIANGLE );
		if ( geom == nullptr )
		{
			const RTCError err = rtcGetDeviceError( m_rtcDevice );
			std::cerr << "[CpuContext] rtcNewGeometry failed at index " << index
					  << ", RTCError=" << static_cast<int>( err ) << std::endl;
			throw std::runtime_error( "rtcNewGeometry failed" );
		}

		rtcSetSharedGeometryBuffer(
			geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
			mesh.vertices, 0, mesh.vertexStride, mesh.vertexCount );

		if ( mesh.triangleIndices != nullptr )
		{
			rtcSetSharedGeometryBuffer(
				geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
				mesh.triangleIndices, 0, mesh.triangleStride, mesh.triangleCount );
		}
		else
		{
			const uint32_t triCount = mesh.triangleCount != 0 ? mesh.triangleCount : ( mesh.vertexCount / 3 );
			uint32_t*	   indices	= static_cast<uint32_t*>( rtcSetNewGeometryBuffer(
				 geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof( uint32_t ), triCount ) );
			if ( indices == nullptr )
			{
				const RTCError err = rtcGetDeviceError( m_rtcDevice );
				rtcReleaseGeometry( geom );
				std::cerr << "[CpuContext] rtcSetNewGeometryBuffer failed at index " << index
						  << ", RTCError=" << static_cast<int>( err ) << std::endl;
				throw std::runtime_error( "rtcSetNewGeometryBuffer failed" );
			}
			for ( uint32_t t = 0; t < triCount; ++t )
			{
				indices[3 * t + 0] = 3 * t + 0;
				indices[3 * t + 1] = 3 * t + 1;
				indices[3 * t + 2] = 3 * t + 2;
			}
		}

		rtcCommitGeometry( geom );
		rtcAttachGeometry( data->rtcScene, geom );
		rtcReleaseGeometry( geom );
	}
	else if ( input.type == hiprtPrimitiveTypeAABBList )
	{
		std::cerr << "[CpuContext] Warning: AABB geometry build is not implemented yet." << std::endl;
	}
	else
	{
		throw std::runtime_error( "CpuContext::buildGeometryEntry: unknown hiprtPrimitiveType" );
	}

	rtcCommitScene( data->rtcScene );
}

void CpuContext::buildGeometries(
	const std::vector<hiprtGeometryBuildInput>& buildInputs,
	const hiprtBuildOptions						buildOptions,
	hiprtDevicePtr								temporaryBuffer,
	oroStream									stream,
	std::vector<hiprtDevicePtr>&				buffers )
{
	(void)buildOptions;
	(void)temporaryBuffer;
	(void)stream;

	for ( size_t i = 0; i < buildInputs.size(); ++i )
		buildGeometryEntry( getCpuGeom( reinterpret_cast<hiprtGeometry>( buffers[i] ) ), buildInputs[i], i );
}

void CpuContext::updateGeometries(
	const std::vector<hiprtGeometryBuildInput>& buildInputs,
	const hiprtBuildOptions						buildOptions,
	hiprtDevicePtr								temporaryBuffer,
	oroStream									stream,
	std::vector<hiprtDevicePtr>&				buffers )
{
	(void)buildOptions;
	(void)temporaryBuffer;
	(void)stream;

	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		CpuGeometryData* data = getCpuGeom( reinterpret_cast<hiprtGeometry>( buffers[i] ) );
		if ( data == nullptr )
			throw std::runtime_error( "CpuContext::updateGeometries: invalid geometry handle" );

		if ( data->rtcScene != nullptr )
			rtcReleaseScene( data->rtcScene );

		data->rtcScene = rtcNewScene( m_rtcDevice );
		if ( data->rtcScene == nullptr )
		{
			const RTCError err = rtcGetDeviceError( m_rtcDevice );
			std::cerr << "[CpuContext] rtcNewScene failed at index " << i
					  << ", RTCError=" << static_cast<int>( err ) << std::endl;
			throw std::runtime_error( "rtcNewScene failed" );
		}

		buildGeometryEntry( data, buildInputs[i], i );
	}
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
