#include "cinder/app/AppBasic.h"
#include "cinder/gl/gl.h"
#include "cinder/gl/Texture.h"

#include "CinderOpenCv.h"
#include "cinder/Capture.h"

#include "BlobResult.h"
#include "Blob.h"

#include "cvblob.h"

using namespace cvb;

using namespace ci;
using namespace ci::app;
using namespace std;

class SimpleBlobDetectionApp : public AppBasic {
public:
	void setup();
	void mouseMove( MouseEvent event );	
	void update();
	void draw();
    void drawBlobs(const CvBlobs& pBlobs);
    
private:  
    Capture			mCap;
    gl::Texture		mTexture;
    CvBlobs         blobs;
    CvTracks        tracks;
    
    int             mThreshold;
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
    
    mThreshold = 100;
}

void SimpleBlobDetectionApp::mouseMove( MouseEvent event )
{
    mThreshold = max(min(event.getX(), 254), 0);
}

void SimpleBlobDetectionApp::update()
{    
    if ( mCap && mCap.checkNewFrame() ) {
        /* get image from capture device */
        cv::Mat input( toOcv( mCap.getSurface() ) );
        
        /* display results */
        mTexture = gl::Texture( mCap.getSurface() );
        
        /* make input image available for opencv */
        IplImage mInputImage = input;
        IplImage* img = &mInputImage;
        
        cvSetImageROI(img, cvRect(0, 0, 640, 480));
        
        /* convert to greyscale */
        IplImage* grey = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
        cvCvtColor(img, grey, CV_BGR2GRAY);
        cvThreshold(grey, grey, mThreshold, 255, CV_THRESH_BINARY);

        cv::Mat mTex = grey;
        mTexture = gl::Texture( fromOcv(mTex) );
        
        /* track blobs */
        IplImage *labelImg = cvCreateImage(cvGetSize(grey), IPL_DEPTH_LABEL,1);
        
        unsigned int result = cvLabel(grey, labelImg, blobs);
        if (!result) {
            console() << "### problem tracking blobs." << endl;            
        }
        
        cvFilterByArea(blobs, 5000, 50000);
        cvUpdateTracks(blobs, tracks, 20., 30, 30);
        
        /* clean up */
        cvReleaseImage(&grey);
        cvReleaseImage(&labelImg);
//        cvReleaseImage(&img);
        
//        cvReleaseBlobs(blobs);
//        cvReleaseTracks(tracks);
    }
}

void SimpleBlobDetectionApp::draw() 
{
    gl::clear();
	if( mTexture ) {
        gl::color(1, 1, 1, 1);
		gl::draw( mTexture );
    }
    
    drawBlobs(blobs);
}

void SimpleBlobDetectionApp::drawBlobs(const CvBlobs& pBlobs)
{
    /* iterate results */
    for (CvBlobs::const_iterator it=blobs.begin(); it!=blobs.end(); ++it) {           
        
        /* draw polygons */
        CvContourPolygon* polygon = cvConvertChainCodesToPolygon(&(*it).second->contour);
        gl::color(1, 0, 0, 1);
        for (int i=0; i<polygon->size(); i++) {
            CvPoint pointA = (*polygon)[i];
            CvPoint pointB = (*polygon)[(i + 1) % polygon->size()];
            gl::drawLine(Vec2f(pointA.x, pointA.y), Vec2f(pointB.x, pointB.y));
        }
        
        /* draw simplified polygons */
        CvContourPolygon* sPolygon = cvSimplifyPolygon(polygon, 10.);
        gl::color(0, 1, 0, 1);
        for (int i=0; i<sPolygon->size(); i++) {
            CvPoint pointA = (*sPolygon)[i];
            CvPoint pointB = (*sPolygon)[(i + 1) % sPolygon->size()];
            gl::drawLine(Vec2f(pointA.x, pointA.y), Vec2f(pointB.x, pointB.y));
        }
        
        /* draw contours */
        CvContourPolygon* cPolygon = cvPolygonContourConvexHull(sPolygon);
        gl::color(0, 0, 1, 1);
        for (int i=0; i<cPolygon->size(); i++) {
            CvPoint pointA = (*cPolygon)[i];
            CvPoint pointB = (*cPolygon)[(i + 1) % cPolygon->size()];
            gl::drawLine(Vec2f(pointA.x, pointA.y), Vec2f(pointB.x, pointB.y));
        }
        
        delete polygon;
        delete sPolygon;
        delete cPolygon;
        
        /* draw internal contours */
        CvContoursChainCode mInternalContours = (*it).second->internalContours;
        for (CvContoursChainCode::iterator mIterator = mInternalContours.begin(); mIterator != mInternalContours.end(); ++mIterator) {
            CvContourChainCode* mInteralContour = *mIterator;
            CvContourPolygon* mInternalPolygon = cvConvertChainCodesToPolygon(mInteralContour);
            gl::color(1, 0, 1, 1);
            for (int i=0; i<mInternalPolygon->size(); i++) {
                CvPoint pointA = (*mInternalPolygon)[i];
                CvPoint pointB = (*mInternalPolygon)[(i + 1) % mInternalPolygon->size()];
                gl::drawLine(Vec2f(pointA.x, pointA.y), Vec2f(pointB.x, pointB.y));
            }
            delete mInternalPolygon;
        }
        
        /* draw tracks */
        gl::color(1, 0.5, 0, 1);
        console() << "### tracks : " << tracks.size() << endl; 
        for (CvTracks::const_iterator it=tracks.begin(); it!=tracks.end(); ++it) {
            CvTrack* mTrack = it->second;
            if (mTrack && !mTrack->inactive) {
                const Rectf & mRect = Rectf(mTrack->minx, mTrack->miny, mTrack->maxx, mTrack->maxy);
                gl::drawStrokedRect(mRect);
            }
        }    
    }
}

CINDER_APP_BASIC( SimpleBlobDetectionApp, RendererGl )
