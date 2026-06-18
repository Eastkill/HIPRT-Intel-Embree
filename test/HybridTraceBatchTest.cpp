#include "HybridTraceBatchTest.h"

#include <hiprt/hiprt_cpu.h>
#include <hiprt/hiprt_hybrid.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

extern const char* g_hip_paths[];
extern const char* g_hiprtc_paths[];

namespace
{
std::filesystem::path getRootDir()
{
	const char* env = std::getenv( "HIPRT_PATH" );
	return env != nullptr ? std::filesystem::path( env ) : std::filesystem::path( ".." );
}

bool readKernelSource( const std::filesystem::path& srcPath, std::string& sourceCode )
{
	std::ifstream f( srcPath );
	if ( !f.is_open() )
		return false;
	sourceCode.assign( std::istreambuf_iterator<char>( f ), std::istreambuf_iterator<char>() );
	return true;
}

hiprtApiFunction buildHybridSceneTraceBatchKernel( hiprtContext context )
{
	const auto		kernelPath = getRootDir() / "hiprt/kernels/HybridSceneTraceBatchKernel.h";
	std::string		sourceCode;
	const std::string moduleName = kernelPath.string();
	if ( !readKernelSource( kernelPath, sourceCode ) )
		return nullptr;

	const char* funcName = "HybridSceneTraceBatchKernel";

	std::vector<std::string> optionStorage;
	optionStorage.push_back( "-I" + getRootDir().string() );
	if ( oroGetCurAPI( 0 ) == ORO_API_HIP )
		optionStorage.push_back( "-ffast-math" );
	else
		optionStorage.push_back( "--use_fast_math" );

	std::vector<const char*> options;
	options.reserve( optionStorage.size() );
	for ( const auto& opt : optionStorage )
		options.push_back( opt.c_str() );

	hiprtApiFunction funcApi = nullptr;
	if ( hiprtBuildTraceKernels(
			 context, 1, &funcName, sourceCode.c_str(), moduleName.c_str(), 0, nullptr, nullptr,
			 static_cast<uint32_t>( options.size() ), options.data(), 0, 0, nullptr, &funcApi, nullptr,
			 true ) != hiprtSuccess )
		return nullptr;

	return funcApi;
}

hiprtFrameSRT makeSrt( float tx, float ty, float tz )
{
	hiprtFrameSRT f{};
	f.rotation	  = { 0.0f, 0.0f, 0.0f, 0.0f };
	f.scale		  = { 1.0f, 1.0f, 1.0f };
	f.translation = { tx, ty, tz };
	f.time		  = 0.0f;
	return f;
}

bool hitsEqual( const hiprtHit& a, const hiprtHit& b )
{
	if ( a.hasHit() != b.hasHit() )
		return false;
	if ( !a.hasHit() )
		return true;
	if ( a.primID != b.primID || a.instanceID != b.instanceID )
		return false;
	return std::fabs( a.t - b.t ) < 1e-4f;
}

void fillTestRays( hiprtRay* rays, uint32_t count )
{
	for ( uint32_t i = 0; i < count; ++i )
	{
		hiprtRay& ray = rays[i];
		ray.minT	  = 0.0f;
		ray.maxT	  = 1e30f;
		ray.direction = { 0.0f, 0.0f, 1.0f };

		if ( i % 2 == 0 )
			ray.origin = { 0.25f, 0.25f, -1.0f };
		else
			ray.origin = { 5.0f, 5.0f, -1.0f };
	}
}
} // namespace

hiprtScene HybridTraceBatchTest::buildSingleInstanceScene( hiprtGeometry& geomOut )
{
	std::vector<hiprtFloat3> vertices;
	std::vector<uint32_t>	 indices;
	hiprtGeometry			 geom = buildTriangleGeometry( vertices, indices );
	if ( geom == nullptr )
		return nullptr;

	std::vector<hiprtInstance> instances( 1 );
	instances[0].type	  = hiprtInstanceTypeGeometry;
	instances[0].geometry = geom;

	std::vector<hiprtFrameSRT> frames = { makeSrt( 0.0f, 0.0f, 0.0f ) };

	hiprtSceneBuildInput in{};
	stageSceneBuildInput( in, instances, frames );

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	hiprtScene scene = nullptr;
	if ( hiprtCreateScene( m_context, in, opts, scene ) != hiprtSuccess )
		return nullptr;
	if ( hiprtBuildScene( m_context, hiprtBuildOperationBuild, in, opts, allocSceneBuildTemp( in, opts ), nullptr, scene ) !=
		 hiprtSuccess )
		return nullptr;

	geomOut = geom;
	return scene;
}

TEST_F( HybridTraceBatchTest, SplitBatchMatchesCpuReference )
{
	constexpr uint32_t rayCount = 16;

	hiprtGeometry geom = nullptr;
	hiprtScene	  scene = buildSingleInstanceScene( geom );
	ASSERT_TRUE( scene != nullptr );
	ASSERT_TRUE( geom != nullptr );

	hiprtApiFunction traceKernel = buildHybridSceneTraceBatchKernel( m_context );
	ASSERT_TRUE( traceKernel != nullptr ) << "Failed to compile HybridSceneTraceBatchKernel.";

	hiprtRay* rays = nullptr;
	hiprtHit* refHits = nullptr;
	hiprtHit* hybridHits = nullptr;
	ASSERT_EQ(
		oroMallocManaged( reinterpret_cast<oroDeviceptr*>( &rays ), rayCount * sizeof( hiprtRay ), oroMemAttachGlobal ),
		oroSuccess );
	ASSERT_EQ(
		oroMallocManaged(
			reinterpret_cast<oroDeviceptr*>( &refHits ), rayCount * sizeof( hiprtHit ), oroMemAttachGlobal ),
		oroSuccess );
	ASSERT_EQ(
		oroMallocManaged(
			reinterpret_cast<oroDeviceptr*>( &hybridHits ), rayCount * sizeof( hiprtHit ), oroMemAttachGlobal ),
		oroSuccess );
	m_managedAllocs.push_back( rays );
	m_managedAllocs.push_back( refHits );
	m_managedAllocs.push_back( hybridHits );

	fillTestRays( rays, rayCount );
	std::memset( refHits, 0, rayCount * sizeof( hiprtHit ) );
	std::memset( hybridHits, 0, rayCount * sizeof( hiprtHit ) );

	hiprtGeomTraversalClosestCPU::traceBatch( m_context, scene, rays, refHits, rayCount );

	hiprtHybridTraceConfig cfg{};
	cfg.gpuFraction = 0.5f;
	cfg.minCpuBatch = 1u;

	hiprtHybridTraceGpuInput gpuInput{};
	gpuInput.traceKernel = traceKernel;

	ASSERT_EQ(
		hiprtTraceHybridClosest( m_context, scene, cfg, gpuInput, rays, hybridHits, rayCount, nullptr ),
		hiprtSuccess );

	for ( uint32_t i = 0; i < rayCount; ++i )
	{
		EXPECT_TRUE( hitsEqual( hybridHits[i], refHits[i] ) )
			<< "Ray " << i << " mismatch (even=hit, odd=miss).";
	}

	ASSERT_EQ( hiprtDestroyScene( m_context, scene ), hiprtSuccess );
	ASSERT_EQ( hiprtDestroyGeometry( m_context, geom ), hiprtSuccess );
}

TEST_F( HybridTraceBatchTest, RejectsNonHybridContext )
{
	hiprtContextCreationInput cpuInput{};
	cpuInput.deviceType	   = hiprtDeviceCPU;
	cpuInput.numCpuThreads = 1;

	hiprtContext cpuContext = nullptr;
	ASSERT_EQ( hiprtCreateContext( HIPRT_API_VERSION, cpuInput, cpuContext ), hiprtSuccess );

	hiprtRay  ray{};
	hiprtHit  hit{};
	hiprtHybridTraceConfig	 cfg{};
	hiprtHybridTraceGpuInput gpuInput{ nullptr, {} };

	EXPECT_EQ(
		hiprtTraceHybridClosest( cpuContext, nullptr, cfg, gpuInput, &ray, &hit, 1, nullptr ),
		hiprtErrorInvalidParameter );

	EXPECT_EQ( hiprtDestroyContext( cpuContext ), hiprtSuccess );
}
