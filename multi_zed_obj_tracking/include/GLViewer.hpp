﻿#ifndef __VIEWER_INCLUDE__
#define __VIEWER_INCLUDE__

#include <random>

#include <sl/Camera.hpp>
#include <sl/Fusion.hpp>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#ifndef M_PI
#define M_PI 3.141592653f
#endif

#define MOUSE_R_SENSITIVITY 0.03f
#define MOUSE_UZ_SENSITIVITY 0.75f
#define MOUSE_DZ_SENSITIVITY 1.25f
#define MOUSE_T_SENSITIVITY 0.05f
#define KEY_T_SENSITIVITY 0.1f

///////////////////////////////////////////////////////////////////////////////////////////////

class Shader
{
public:
    Shader()
    {
    }
    Shader(GLchar *vs, GLchar *fs);
    ~Shader();
    GLuint getProgramId();

    static const GLint ATTRIB_VERTICES_POS = 0;
    static const GLint ATTRIB_COLOR_POS = 1;
    static const GLint ATTRIB_NORMAL = 2;

private:
    bool compile(GLuint &shaderId, GLenum type, GLchar *src);
    GLuint verterxId_;
    GLuint fragmentId_;
    GLuint programId_;
};

struct ShaderData
{
    Shader it;
    GLuint MVP_Mat;
};

class Simple3DObject {
public:

    Simple3DObject();
    Simple3DObject(sl::Translation position, bool isStatic);
    ~Simple3DObject();

    void addBBox(std::vector<sl::float3> &pts, sl::float4 clr);
    void addPoint(sl::float3 pt, sl::float4 clr);
    void addTriangle(sl::float3 p1, sl::float3 p2, sl::float3 p3, sl::float4 clr);
    void addLine(sl::float3 p1, sl::float3 p2, sl::float4 clr);
    void addFace(sl::float3 p1, sl::float3 p2, sl::float3 p3, sl::float4 clr);

    // New 3D rendering
    void addFullEdges(std::vector<sl::float3> &pts, sl::float4 clr);
    void addVerticalEdges(std::vector<sl::float3> &pts, sl::float4 clr);
    void addTopFace(std::vector<sl::float3> &pts, sl::float4 clr);
    void addVerticalFaces(std::vector<sl::float3> &pts, sl::float4 clr);

    // Sphere
    void createSphere(sl::float3 origin);

    void pushToGPU();
    void clear();

    void setDrawingType(GLenum type);

    void draw();

    void translate(const sl::Translation& t);
    void setPosition(const sl::Translation& p);

    void setRT(const sl::Transform& mRT);

    void rotate(const sl::Orientation& rot);
    void rotate(const sl::Rotation& m);
    void setRotation(const sl::Orientation& rot);
    void setRotation(const sl::Rotation& m);

    const sl::Translation& getPosition() const;

    sl::Transform getModelMatrix() const;
private:
    std::vector<sl::float3> vertices_;
    std::vector<sl::float4> colors_;
    std::vector<unsigned int> indices_;

    bool isStatic_;
    bool need_update;
    GLenum drawingType_;

    GLuint vaoID_;
    /*
    Vertex buffer IDs:
    - [0]: Vertices coordinates;
    - [1]: Colors;
    - [2]: Indices;
     */
    GLuint vboID_[3];

    sl::Translation position_;
    sl::Orientation rotation_;
};

class CustomObjectDetection
{
public:
    CustomObjectDetection();
    ~CustomObjectDetection();

    // Initialize Opengl and Cuda buffers
    // Warning: must be called in the Opengl thread
    void initialize(sl::Objects &, sl::float4 clr);
    void initialize(sl::float3 origin, sl::float3 dim, sl::float4 clr);
    // Push a new custom object detection
    // Warning: can be called from any thread but the mutex "mutexData" must be locked
    void pushNewOD();
    // Draw the object detection bounding box
    // Warning: must be called in the Opengl thread
    void draw(const sl::Transform &vp);
    // Close (disable update)
    void close();

    Simple3DObject BBox_edges;
    Simple3DObject BBox_faces;

private:
    
    std::vector<sl::ObjectData> objs;
    sl::float4 clr;
    void createBboxRendering(std::vector<sl::float3> &, sl::float4);
    bool isInit = false;

    ShaderData shader;
};

class CameraGL
{
public:
    CameraGL()
    {
    }

    enum DIRECTION
    {
        UP,
        DOWN,
        LEFT,
        RIGHT,
        FORWARD,
        BACK
    };
    CameraGL(sl::Translation position, sl::Translation direction, sl::Translation vertical = sl::Translation(0, 1, 0)); // vertical = Eigen::Vector3f(0, 1, 0)
    ~CameraGL();

    void update();
    void setProjection(float horizontalFOV, float verticalFOV, float znear, float zfar);
    const sl::Transform &getViewProjectionMatrix() const;

    float getHorizontalFOV() const;
    float getVerticalFOV() const;

    // Set an offset between the eye of the camera and its position
    // Note: Useful to use the camera as a trackball camera with z>0 and x = 0, y = 0
    // Note: coordinates are in local space
    void setOffsetFromPosition(const sl::Translation &offset);
    const sl::Translation &getOffsetFromPosition() const;

    void setDirection(const sl::Translation &direction, const sl::Translation &vertical);
    void translate(const sl::Translation &t);
    void setPosition(const sl::Translation &p);
    void rotate(const sl::Orientation &rot);
    void rotate(const sl::Rotation &m);
    void setRotation(const sl::Orientation &rot);
    void setRotation(const sl::Rotation &m);

    const sl::Translation &getPosition() const;
    const sl::Translation &getForward() const;
    const sl::Translation &getRight() const;
    const sl::Translation &getUp() const;
    const sl::Translation &getVertical() const;
    float getZNear() const;
    float getZFar() const;

    static const sl::Translation ORIGINAL_FORWARD;
    static const sl::Translation ORIGINAL_UP;
    static const sl::Translation ORIGINAL_RIGHT;

    sl::Transform projection_;
    bool usePerspective_;

private:
    void updateVectors();
    void updateView();
    void updateVPMatrix();

    sl::Translation offset_;
    sl::Translation position_;
    sl::Translation forward_;
    sl::Translation up_;
    sl::Translation right_;
    sl::Translation vertical_;

    sl::Orientation rotation_;

    sl::Transform view_;
    sl::Transform vpMatrix_;
    float horizontalFieldOfView_;
    float verticalFieldOfView_;
    float znear_;
    float zfar_;
};

class PointCloud
{
public:
    PointCloud();
    ~PointCloud();

    // Initialize Opengl and Cuda buffers
    // Warning: must be called in the Opengl thread
    void initialize(sl::Mat &, sl::float4 clr);
    // Push a new point cloud
    // Warning: can be called from any thread but the mutex "mutexData" must be locked
    void pushNewPC();
    // Draw the point cloud
    // Warning: must be called in the Opengl thread
    void draw(const sl::Transform &vp, bool draw_clr);
    // Close (disable update)
    void close();

private:
    sl::Mat refMat;
    sl::float4 clr;

    Shader shader_;
    GLuint shMVPMatrixLoc_;
    GLuint shDrawColor;
    GLuint shColor;

    size_t numBytes_;
    float *xyzrgbaMappedBuf_;
    GLuint bufferGLID_;
    cudaGraphicsResource *bufferCudaID_;
};

class CameraViewer
{
public:
    CameraViewer();
    ~CameraViewer();

    // Initialize Opengl and Cuda buffers
    bool initialize(sl::Mat &image, sl::float4 clr);
    // Push a new Image + Z buffer and transform into a point cloud
    void pushNewImage();
    // Draw the Image
    void draw(sl::Transform vpMatrix);
    // Close (disable update)
    void close();

    Simple3DObject frustum;

private:
    sl::Mat ref;
    cudaArray_t ArrIm;
    cudaGraphicsResource *cuda_gl_ressource; // cuda GL resource
    Shader shader;
    GLuint shMVPMatrixLocTex_;

    GLuint texture;
    GLuint vaoID_;
    GLuint vboID_[3];

    std::vector<sl::uint3> faces;
    std::vector<sl::float3> vert;
    std::vector<sl::float2> uv;
};

struct ObjectClassName
{
    sl::float3 position;
    std::string name_lineA;
    std::string name_lineB;
    sl::float4 color;
};

struct ZedRenderData
{
    int id;
    sl::Mat views;
    sl::Objects objs;
    sl::Mat pcs;

    ZedRenderData(int _id, const sl::Mat& _views, const sl::Objects& _objs, const sl::Mat& _pcs) 
        : id(_id), views(_views), objs(_objs), pcs(_pcs) {}
};

// This class manages input events, window and Opengl rendering pipeline

class GLViewer
{
public:
    GLViewer();
    ~GLViewer();
    bool isAvailable();
    void init(int argc, char **argv);

    void updateMultiCamera(int, sl::Mat &, sl::Objects &, sl::Mat &);

    void updateFusion(std::vector<ZedRenderData> &, std::map<int, sl::Transform> &);
    
    void setCameraPose(int, sl::Transform);

    unsigned char getKey()
    {
        auto ret_v = lastPressedKey;
        lastPressedKey = ' ';
        return ret_v;
    }

    void exit();

private:
    void render();
    void update();
    void draw();
    void clearInputs();
    void setRenderCameraProjection(sl::CameraParameters params, float znear, float zfar);

    void printText();

    // Glut functions callbacks
    static void drawCallback();
    static void mouseButtonCallback(int button, int state, int x, int y);
    static void mouseMotionCallback(int x, int y);
    static void reshapeCallback(int width, int height);
    static void keyPressedCallback(unsigned char c, int x, int y);
    static void keyReleasedCallback(unsigned char c, int x, int y);
    static void idle();

    bool available;
    bool drawBbox = false;
    bool hasDetection = false;

    enum MOUSE_BUTTON
    {
        LEFT = 0,
        MIDDLE = 1,
        RIGHT = 2,
        WHEEL_UP = 3,
        WHEEL_DOWN = 4
    };

    enum KEY_STATE
    {
        UP = 'u',
        DOWN = 'd',
        FREE = 'f'
    };

    unsigned char lastPressedKey;

    bool mouseButton_[3];
    int mouseWheelPosition_;
    int mouseCurrentPosition_[2];
    int mouseMotion_[2];
    int previousMouseMotion_[2];
    KEY_STATE keyStates_[256];

    std::mutex mtx;

    ShaderData shader;

    sl::Transform projection_;
    sl::float4 bckgrnd_clr;

    std::map<int, PointCloud> point_clouds;
    std::map<int, CameraViewer> viewers;
    std::map<int, sl::Transform> poses;
    std::map<int, CustomObjectDetection> detections;

    std::map<int, sl::float4> colors;

    std::vector<ObjectClassName> fusionStats;

    CameraGL camera_;
    Simple3DObject floor_grid;
    CustomObjectDetection fused_detections;

    bool show_object = true;
    bool show_fused_object = true;
    bool show_pc = true;
    bool draw_flat_color = false;
    
    std::uniform_int_distribution<uint32_t> uint_dist360;
    std::mt19937 rng;
};

#endif /* __VIEWER_INCLUDE__ */
