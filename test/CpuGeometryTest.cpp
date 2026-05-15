#include "CpuGeometryTest.h"

#include <hiprt/impl/CpuTypes.h>
#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>

void CpuGeometryTest::SetUp()
{
	hiprtContextCreationInput ctxInput{};
	ctxInput.deviceType    = hiprtDeviceCPU;
	ctxInput.numCpuThreads = 1;

	ASSERT_EQ( hiprtCreateContext( HIPRT_API_VERSION, ctxInput, m_context ), hiprtSuccess )
		<< "Failed to create CPU context. Ensure hiprt.cpp routing is working.";
	ASSERT_TRUE( m_context != nullptr ) << "Context pointer is null after creation.";
}

void CpuGeometryTest::TearDown()
{
	if ( m_context != nullptr )
		EXPECT_EQ( hiprtDestroyContext( m_context ), hiprtSuccess )
			<< "Failed to cleanly destroy CPU context.";
}

hiprtGeometry CpuGeometryTest::buildSampleTriangle(
	std::vector<hiprtFloat3>&	vertices,
	std::vector<uint32_t>&		indices,
	hiprtGeometryBuildInput&	geomInputOut )
{
	vertices = {
		{ 0.0f, 0.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f } };
	indices = { 0, 1, 2 };

	hiprtTriangleMeshPrimitive mesh{};
	mesh.vertices        = vertices.data();
	mesh.vertexCount     = 3;
	mesh.vertexStride    = sizeof( hiprtFloat3 );
	mesh.triangleIndices = indices.data();
	mesh.triangleCount   = 1;
	mesh.triangleStride  = 3 * sizeof( uint32_t );

	geomInputOut							 = {};
	geomInputOut.type						 = hiprtPrimitiveTypeTriangleMesh;
	geomInputOut.primitive.triangleMesh		 = mesh;

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	hiprtGeometry geom = nullptr;
	EXPECT_EQ( hiprtCreateGeometry( m_context, geomInputOut, opts, geom ), hiprtSuccess )
		<< "hiprtCreateGeometry failed.";
	EXPECT_TRUE( geom != nullptr ) << "Returned geometry handle is null.";
	EXPECT_EQ(
		hiprtBuildGeometry( m_context, hiprtBuildOperationBuild, geomInputOut, opts, nullptr, nullptr, geom ),
		hiprtSuccess )
		<< "hiprtBuildGeometry (Build) failed.";
	return geom;
}

namespace
{
bool traceHit( hiprtGeometry geom, float ox, float oy, float oz, float dx, float dy, float dz )
{
	auto* data = reinterpret_cast<hiprt::CpuGeometryData*>( geom );
	if ( data == nullptr || data->rtcScene == nullptr )
		return false;

	RTCRayHit rh{};
	rh.ray.org_x  = ox;
	rh.ray.org_y  = oy;
	rh.ray.org_z  = oz;
	rh.ray.dir_x  = dx;
	rh.ray.dir_y  = dy;
	rh.ray.dir_z  = dz;
	rh.ray.tnear  = 0.0f;
	rh.ray.tfar   = 1e30f;
	rh.ray.mask   = static_cast<unsigned>( -1 );
	rh.ray.flags  = 0;
	rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
	rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

	rtcIntersect1( data->rtcScene, &rh );
	return rh.hit.geomID != RTC_INVALID_GEOMETRY_ID;
}
} // namespace

TEST_F( CpuGeometryTest, CreateAndDestroyGeometry )
{
	std::vector<hiprtFloat3> vertices = {
		{ 0.0f, 0.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f } };
	std::vector<uint32_t> indices = { 0, 1, 2 };

	hiprtTriangleMeshPrimitive mesh{};
	mesh.vertices        = vertices.data();
	mesh.vertexCount     = 3;
	mesh.vertexStride    = sizeof( hiprtFloat3 );
	mesh.triangleIndices = indices.data();
	mesh.triangleCount   = 1;
	mesh.triangleStride  = 3 * sizeof( uint32_t );

	hiprtGeometryBuildInput in{};
	in.type                   = hiprtPrimitiveTypeTriangleMesh;
	in.primitive.triangleMesh = mesh;

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	hiprtGeometry geom = nullptr;
	ASSERT_EQ( hiprtCreateGeometry( m_context, in, opts, geom ), hiprtSuccess )
		<< "hiprtCreateGeometry failed on CPU context.";
	ASSERT_TRUE( geom != nullptr ) << "Returned geometry handle is null.";
	ASSERT_EQ( hiprtDestroyGeometry( m_context, geom ), hiprtSuccess )
		<< "hiprtDestroyGeometry failed to release Embree scene memory.";
}

TEST_F( CpuGeometryTest, BuildGeometryProducesHittableBvh )
{
	std::vector<hiprtFloat3>	vertices;
	std::vector<uint32_t>		indices;
	hiprtGeometryBuildInput		in{};
	hiprtGeometry				geom = buildSampleTriangle( vertices, indices, in );
	ASSERT_TRUE( geom != nullptr );

	EXPECT_TRUE( traceHit( geom, 0.25f, 0.25f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Promien ze srodka trojkata powinien trafic w zbudowane BVH.";
	EXPECT_FALSE( traceHit( geom, 2.0f, 2.0f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Promien poza trojkatem nie powinien trafiac.";

	ASSERT_EQ( hiprtDestroyGeometry( m_context, geom ), hiprtSuccess );
}

TEST_F( CpuGeometryTest, UpdateGeometryRebuildsBvh )
{
	std::vector<hiprtFloat3>	vertices;
	std::vector<uint32_t>		indices;
	hiprtGeometryBuildInput		in{};
	hiprtGeometry				geom = buildSampleTriangle( vertices, indices, in );
	ASSERT_TRUE( geom != nullptr );

	EXPECT_FALSE( traceHit( geom, 0.1f, 1.5f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Przed update BVH nie powinno obejmowac y=1.5.";

	vertices[2]							 = { 0.0f, 2.0f, 0.0f };
	in.primitive.triangleMesh.vertices	 = vertices.data();

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	ASSERT_EQ(
		hiprtBuildGeometry( m_context, hiprtBuildOperationUpdate, in, opts, nullptr, nullptr, geom ),
		hiprtSuccess )
		<< "hiprtBuildGeometry (Update) failed to rebuild the scene.";

	EXPECT_TRUE( traceHit( geom, 0.1f, 1.5f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Po update BVH powinno obejmowac y=1.5 (nowy wierzcholek y=2).";

	ASSERT_EQ( hiprtDestroyGeometry( m_context, geom ), hiprtSuccess );
}
