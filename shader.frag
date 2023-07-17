//============================================================================
// GROUP NUMBER: 12 
//
// STUDENT NAME: HU ZHANPENG
// NUS User ID.: t0930152
//
// STUDENT NAME: NING JUNTING
// NUS User ID.: t0930085
//
// STUDENT NAME: LUO TANGWEN
// NUS User ID.: t0930258
//
// COMMENTS TO GRADER: 
//
//============================================================================

// FRAGMENT SHADER

#version 330 core

//============================================================================
// Received from rasterizer.
//============================================================================
in vec3 ecPosition;   // Fragment's 3D position in eye space.
in vec3 ecNormal;     // Fragment's normal vector in eye space.
in vec3 v2fTexCoord;  // Fragment's texture coordinates. It is 3D when it is
//============================================================================
// Indicates which object is being rendered.
// 0 -- draw skybox, 1 -- draw brick cube, 2 -- draw wooden cube.
//============================================================================
uniform int WhichObj;

//============================================================================
// View and projection matrices, etc.
//============================================================================
uniform mat4 ViewMatrix;          // View transformation matrix.
uniform mat4 ModelViewMatrix;     // ModelView matrix.
uniform mat4 ModelViewProjMatrix; // ModelView matrix * Projection matrix.
uniform mat3 NormalMatrix;        // For transforming object-space direction
                                  //   vector to eye space.

//============================================================================
// Light info.
//============================================================================
uniform vec4 LightPosition; // Given in eye space. Can be directional.
uniform vec4 LightAmbient;
uniform vec4 LightDiffuse;
uniform vec4 LightSpecular;

//============================================================================
// Material info.
//============================================================================
// Specular material of the brick surface.
const vec3 BrickSpecularMaterial = vec3(1.0, 1.0, 1.0);

// Material shininess of specular reflection on the brick surface.
const float BrickShininess = 128.0;

// Specular material of the wood surface.
const vec3 WoodSpecularMaterial = vec3(1.0, 1.0, 1.0);

// Material shininess of specular reflection on the wood surface.
const float WoodShininess = 128.0;

//============================================================================
// Environment cubemap used for skybox and reflection mapping.
//============================================================================
uniform samplerCube EnvMap;

//============================================================================
// The brick texture map whose color is used as the ambient and diffuse
// material in the lighting computation.
//============================================================================
uniform sampler2D BrickDiffuseMap;

//============================================================================
// The brick normal map whose color is used as perturbed normal vector
// in the tangent space.
//============================================================================
uniform sampler2D BrickNormalMap;

//============================================================================
// The wood texture map whose color is used as the ambient and diffuse
// material in the lighting computation.
//============================================================================
uniform sampler2D WoodDiffuseMap;

//============================================================================
// MirrorTileDensity defines the number of hemispherical mirrors across each
// dimension when the corresponding texture coordinate ranges from 0.0 to 1.0.
//============================================================================
const float MirrorTileDensity = 2.0;  // (0.0, inf)

//============================================================================
// MirrorRadius is the radius of the hemispherical mirror in each tile.
// The radius is relative to the tile size, which is considered to be 1.0 x 1.0.
//============================================================================
const float MirrorRadius = 0.4;  // (0.0, 0.5]

//============================================================================
// DeltaNormal_Z_Scale is used to exaggerate the height of bump when doing
// normal mapping. The z component of the decoded perturbed normal vector
// read from the normal map is multiplied by DeltaNormal_Z_Adj.
//============================================================================
const float DeltaNormal_Z_Scale = 1.0 / 5.0;

//============================================================================
// Output to color buffer.
//============================================================================
layout (location = 0) out vec4 FragColor;



/////////////////////////////////////////////////////////////////////////////
// Compute fragment color on skybox.
/////////////////////////////////////////////////////////////////////////////
void drawSkybox()
{
    FragColor = texture(EnvMap, v2fTexCoord);
}



/////////////////////////////////////////////////////////////////////////////
// Compute the Tangent vector T and the Binormal vector B, given the
// Normal vector N, a 3D position p, and 2D texture coordinates uv.
// Note that T, B, N and p are all in the same coordinate space.
/////////////////////////////////////////////////////////////////////////////
void compute_tangent_vectors( vec3 N, vec3 p, vec2 uv, out vec3 T, out vec3 B )
{
    // Please refer to "Followup: Normal Mapping Without Precomputed Tangents" at
    // http://www.thetenthplanet.de/archives/1180

    // get edge vectors of the pixel triangle
    vec3 dp1 = dFdx( p );
    vec3 dp2 = dFdy( p );
    vec2 duv1 = dFdx( uv );
    vec2 duv2 = dFdy( uv );

    // solve the linear system
    vec3 dp2perp = cross( dp2, N );
    vec3 dp1perp = cross( N, dp1 );
    T = normalize( dp2perp * duv1.x + dp1perp * duv2.x );  // Tangent vector
    B = normalize( dp2perp * duv1.y + dp1perp * duv2.y );  // Binormal vector
}



/////////////////////////////////////////////////////////////////////////////
// Compute fragment color on brick cube.
/////////////////////////////////////////////////////////////////////////////
void drawBrickCube()
{
    if (gl_FrontFacing) {
        vec3 viewVec = -normalize(ecPosition);
        vec3 necNormal = normalize(ecNormal);

        vec3 lightVec;
        if (LightPosition.w == 0.0 )
            lightVec = normalize(LightPosition.xyz);
        else
            lightVec = normalize(LightPosition.xyz - ecPosition);

        /////////////////////////////////////////////////////////////////////////////
        // TASK 2:
        // * Construct eye-space Tangent and Binormal vectors.
        // * Read and decode tangent-space perturbation vector from normal map.
        // * Transform perturbation vector to eye space.
        // * Use eye-space perturbation vector as normal vector in lighting
        //   computation using Phong Reflection Model.
        // * Write computed fragment color to FragColor.
        /////////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////
        vec3 nTangent;
        vec3 nBinormal;
        // Construct eye-space Tangent and Binormal vectors
        compute_tangent_vectors(necNormal, viewVec, v2fTexCoord.xy, nTangent, nBinormal);
        // Make sure N, T, B form an orthogonal system
        nTangent = normalize(cross(necNormal, nBinormal));
        nBinormal = normalize(cross(necNormal, nTangent));
        // Read and decode tangent-space perturbation vector from normal map.
        vec3 tanPerturbedNormal = normalize(vec3(texture(BrickNormalMap, v2fTexCoord.xy).rgb * 2.0 - 1.0));

        // Transform perturbation vector to eye space.
        vec3 ecPerturbedNormal = normalize(tanPerturbedNormal.x * nTangent + tanPerturbedNormal.y * nBinormal + DeltaNormal_Z_Scale * tanPerturbedNormal.z * necNormal);
        
        // Use eye-space perturbation vector as normal vector in lighting computation using Phong Reflection Model
        vec3 reflectVec = normalize(reflect(-lightVec, ecPerturbedNormal));
        float N_dot_L = max(0.0, dot(ecPerturbedNormal, lightVec));
        float R_dot_V = max(0.0, dot(reflectVec, viewVec));
        
        vec3 ambient = LightAmbient.rgb * texture(BrickDiffuseMap, v2fTexCoord.xy).rgb;

        vec3 diffuse = LightDiffuse.rgb * texture(BrickDiffuseMap, v2fTexCoord.xy).rgb * N_dot_L;
        
        float spec = (R_dot_V == 0.0) ? 0.0 : pow(R_dot_V, BrickShininess);
        vec3 specular = LightSpecular.rgb * spec * BrickSpecularMaterial;
        
        vec3 result = ambient + diffuse + specular;
        ///////////////////////////////////

        FragColor = vec4(result, 1.0);  // Replace this with your code.
    }
    else discard;
}



/////////////////////////////////////////////////////////////////////////////
// Compute fragment color on wooden cube.
/////////////////////////////////////////////////////////////////////////////
void drawWoodenCube()
{
    if (gl_FrontFacing) {
        vec3 viewVec = -normalize(ecPosition);
        vec3 necNormal = normalize(ecNormal);

        vec3 lightVec;
        if (LightPosition.w == 0.0 )
            lightVec = normalize(LightPosition.xyz);
        else
            lightVec = normalize(LightPosition.xyz - ecPosition);

        /////////////////////////////////////////////////////////////////////////////
        // TASK 3:
        // * Determine whether fragment is in wood region or mirror region.
        // * If fragment is in wood region,
        //    -- Read from wood texture map.
        //    -- Perform Phong lighting computation using the wood texture
        //       color as the ambient and diffuse material.
        //    -- Write computed fragment color to FragColor.
        // * If fragment is in mirror region,
        //    -- Construct eye-space Tangent and Binormal vectors.
        //    -- Construct tangent-space perturbation vector for a
        //       hemispherical bump.
        //    -- Transform perturbation vector to eye space.
        //    -- Reflect the view vector about the eye-space perturbation vector.
        //    -- Transform reflection vector to World Space.
        //    -- Use world-space reflection vector to access environment cubemap.
        //    -- Write computed fragment color to FragColor.
        /////////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////

        vec3 nTangent;
        vec3 nBinormal;
        compute_tangent_vectors(necNormal, viewVec, v2fTexCoord.xy, nTangent, nBinormal);
        // Make sure N, T, B form an orthogonal system
        nTangent = normalize(cross(necNormal, nBinormal));
        nBinormal = normalize(cross(necNormal, nTangent));

        vec2 p = fract(MirrorTileDensity * v2fTexCoord.xy) - 0.5;
        float Dist = sqrt(p.x * p.x + p.y * p.y);
        // Check whether the fragment is in wood region
        if(Dist >= MirrorRadius){
            // Perform Phong lighting computation using the wood texture color as the ambient and diffuse material.
            vec3 reflectVec = normalize(reflect(-lightVec, ecNormal));
            float N_dot_L = max(0.0, dot(ecNormal, lightVec));
            float R_dot_V = max(0.0, dot(reflectVec, viewVec));
            
            vec3 ambient = LightAmbient.rgb * texture(WoodDiffuseMap, v2fTexCoord.xy).rgb;

            vec3 diffuse = LightDiffuse.rgb * texture(WoodDiffuseMap, v2fTexCoord.xy).rgb * N_dot_L;
            
            float spec = (R_dot_V == 0.0) ? 0.0 : pow(R_dot_V, WoodShininess);
            vec3 specular = LightSpecular.rgb * spec * WoodSpecularMaterial;
            

            vec3 result = ambient + diffuse + specular;
            FragColor = vec4(result, 1.0); 
        }
        else{
        
            vec3 tanPerturbedNormal = normalize(vec3(p.x, p.y, sqrt(MirrorRadius * MirrorRadius - p.x * p.x - p.y * p.y)));
            // Transform perturbation vector to eye space.
            vec3 ecPerturbedNormal = normalize(tanPerturbedNormal.x * nTangent + tanPerturbedNormal.y * nBinormal + tanPerturbedNormal.z * necNormal);
           
            // Reflect the view vector about the eye-space perturbation vector.
            vec3 ecreflectView = normalize(reflect(-viewVec, ecPerturbedNormal));

            // Transform reflection vector to World Space.
            mat4 invM = inverse(ViewMatrix);
            vec3 wcreflectView = normalize(vec3(invM * vec4(ecreflectView,0)));

            // Use world-space reflection vector to access environment cubemap.
            vec3 diff = texture(EnvMap, wcreflectView).rgb;
            
            FragColor = vec4(diff, 1.0); 
            
        ///////////////////////////////////
        }
        ///////////////////////////////////

    }
    else discard;
}



void main()
{
    switch(WhichObj) {
        case 0: drawSkybox(); break;
        case 1: drawBrickCube(); break;
        case 2: drawWoodenCube(); break;
    }
}
