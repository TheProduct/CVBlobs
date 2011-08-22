/**
 * BlobTracking DLed from
 * http://opencv.willowgarage.com/wiki/cvBlobsLib#Download
 */

// maybe also interesting http://stackoverflow.com/questions/6174527/emgu-cv-blob-detection or this http://code.google.com/p/opentouch/

#include "cinder/app/AppBasic.h"
#include "cinder/gl/gl.h"
#include "cinder/gl/Texture.h"

#include "CinderOpenCv.h"
#include "cinder/Capture.h"

#include "BlobResult.h"
#include "Blob.h"


using namespace ci;
using namespace ci::app;
using namespace std;

class SimpleBlobDetectionApp : public AppBasic {
public:
	void setup();
	void mouseDown( MouseEvent event );	
	void update();
	void draw();
    
private:
    void drawBlob ( CBlobResult blobs );
    
    Capture			mCap;
    gl::Texture		mTexture;
    CBlobResult     blobs;
    
};

void SimpleBlobDetectionApp::setup()
{
    try {
		mCap = Capture( 640, 480 );
		mCap.start();
	}
	catch( ... ) {
		console() << "Failed to initialize capture" << std::endl;
	}
}

void SimpleBlobDetectionApp::mouseDown( MouseEvent event )
{
}

void SimpleBlobDetectionApp::update()
{    
    if ( mCap && mCap.checkNewFrame() ) {
        /* extract grayscale image */
        cv::Mat input( toOcv( mCap.getSurface() ) );
        mTexture = gl::Texture( fromOcv( input ) );
        cv::Mat mGrayScaleImage;
        extractChannel(input, mGrayScaleImage, 1);
        IplImage mCapturedImage = mGrayScaleImage;
        
        /* threshold image */
        IplImage* tmp = 0;
        tmp = cvCreateImage(cvGetSize(&mCapturedImage), IPL_DEPTH_8U, 1);
        cvThreshold( &mCapturedImage, tmp, 64, 255, CV_THRESH_BINARY );
        
        /* retreive and filter blobs */
        blobs = CBlobResult( tmp, NULL, 64, true );
        blobs.Filter( blobs, B_INCLUDE, CBlobGetArea(), B_GREATER, 1000);
//        blobs.Filter( blobs, B_EXCLUDE, CBlobGetArea(), B_LESS, 3000 );
        
        /* display results */
        cv::Mat mDisplay = tmp;
        mTexture = gl::Texture( fromOcv( mDisplay ) );
        cvReleaseImage(&tmp);
    }
}

void SimpleBlobDetectionApp::draw()
{
	// clear out the window with black
    gl::clear();
	if( mTexture ) {
        gl::color(1, 1, 1, 1);
		gl::draw( mTexture );
    }
    gl::color(1, 0, 0, 1);
    drawBlob(blobs);
}

void SimpleBlobDetectionApp::drawBlob(CBlobResult blobs) {
    CBlob bl;    
    for ( int i = 0; i < blobs.GetNumBlobs(); i++ ) {
        CBlob b = blobs.GetBlob(i);
        
        CvSeq *convexHull = cvConvexHull2(b.Edges());
        for (int i = 0; i < convexHull->total; i++) {
            int mNextPoint = i++ % convexHull->total;
            CvPoint pointA = **CV_GET_SEQ_ELEM( CvPoint*, convexHull, i );
            CvPoint pointB = **CV_GET_SEQ_ELEM( CvPoint*, convexHull, mNextPoint );
            gl::drawLine(Vec2f(pointA.x, pointA.y), Vec2f(pointB.x, pointB.y));
        }
    }
}

CINDER_APP_BASIC( SimpleBlobDetectionApp, RendererGl )
