#include "cinder/app/AppBasic.h"
#include "cinder/gl/gl.h"
#include "cinder/gl/Texture.h"

#include "CinderOpenCv.h"
#include "cinder/Capture.h"

#include "cvblob.h"

using namespace cvb;

using namespace ci;
using namespace ci::app;
using namespace std;

class SimpleBlobDetectionApp : public AppBasic {
public:
	void setup();
	void mouseMove( MouseEvent );	
	void update();
	void draw();
    
private:  
    void drawBlobsAndTracks( const CvBlobs&, const CvTracks& );

    Capture			mCap;
    gl::Texture		mTexture;
    CvBlobs         mBlobs;
    CvTracks        mTracks;
    
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
                
        /* make input image available for opencv */
        IplImage mInputImage = input;
        IplImage* img = &mInputImage;
        
        cvSetImageROI(img, cvRect(0, 0, 640, 480));
        
        /* convert to grey scale */
        IplImage* mGreyImage = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
        cvCvtColor(img, mGreyImage, CV_BGR2GRAY);
        cvThreshold(mGreyImage, mGreyImage, mThreshold, 255, CV_THRESH_BINARY);
        
        /* track blobs */
        cvReleaseBlobs(mBlobs);
        cvReleaseTracks(mTracks);

        IplImage* mLabelImg = cvCreateImage(cvGetSize(mGreyImage), IPL_DEPTH_LABEL,1);
        
        unsigned int result = cvLabel(mGreyImage, mLabelImg, mBlobs);
        if (!result) {
            console() << "### problem tracking blobs." << endl;            
        }
        
        cvFilterByArea(mBlobs, 5000, 50000);
        cvUpdateTracks(mBlobs, mTracks, 5., 10);
                
        /* display results */
        const bool SHOW_ORIGINAL_CAPTURE = true;
        if (SHOW_ORIGINAL_CAPTURE) {
            if (mTexture) {
                mTexture.update( mCap.getSurface() );
            } else {
                mTexture = gl::Texture( mCap.getSurface() );
            }
        } else {
            cv::Mat mTex = mGreyImage;
            if (mTexture) {
                mTexture.update( fromOcv(mTex) );
            } else {
                mTexture = gl::Texture( fromOcv(mTex) );
            }
        }

        /* clean up */
        cvReleaseImage(&mGreyImage);
        cvReleaseImage(&mLabelImg);

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
    
    drawBlobsAndTracks(mBlobs, mTracks);
}

void SimpleBlobDetectionApp::drawBlobsAndTracks(const CvBlobs& pBlobs, const CvTracks& pTracks)
{
    /* iterate results */
    for (CvBlobs::const_iterator it=pBlobs.begin(); it!=pBlobs.end(); ++it) {           
        
        /* draw polygons */
        const CvContourPolygon* polygon = cvConvertChainCodesToPolygon(&(*it).second->contour);
        gl::color(1, 0, 0, 1);
        for (int i=0; i<polygon->size(); i++) {
            const CvPoint pointA = (*polygon)[i];
            const CvPoint pointB = (*polygon)[(i + 1) % polygon->size()];
            gl::drawLine(Vec2f(pointA.x, pointA.y), Vec2f(pointB.x, pointB.y));
        }
        
        /* draw simplified polygons */
        const CvContourPolygon* sPolygon = cvSimplifyPolygon(polygon, 10.);
        gl::color(0, 1, 0, 1);
        for (int i=0; i<sPolygon->size(); i++) {
            const CvPoint pointA = (*sPolygon)[i];
            const CvPoint pointB = (*sPolygon)[(i + 1) % sPolygon->size()];
            gl::drawLine(Vec2f(pointA.x, pointA.y), Vec2f(pointB.x, pointB.y));
        }
        
        /* draw contours */
        const CvContourPolygon* cPolygon = cvPolygonContourConvexHull(sPolygon);
        gl::color(0, 0, 1, 1);
        for (int i=0; i<cPolygon->size(); i++) {
            const CvPoint pointA = (*cPolygon)[i];
            const CvPoint pointB = (*cPolygon)[(i + 1) % cPolygon->size()];
            gl::drawLine(Vec2f(pointA.x, pointA.y), Vec2f(pointB.x, pointB.y));
        }
        
        delete polygon;
        delete sPolygon;
        delete cPolygon;
        
        /* draw internal contours */
        CvContoursChainCode mInternalContours = (*it).second->internalContours;
        for (CvContoursChainCode::iterator mIterator = mInternalContours.begin(); mIterator != mInternalContours.end(); ++mIterator) {
            const CvContourChainCode* mInteralContour = *mIterator;
            const CvContourPolygon* mInternalPolygon = cvConvertChainCodesToPolygon(mInteralContour);
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
//        console() << "### tracks : " << pTracks.size() << endl; 
        for (CvTracks::const_iterator it=pTracks.begin(); it!=pTracks.end(); ++it) {
            const CvTrack* mTrack = it->second;
            if (mTrack && !mTrack->inactive) {
                const Rectf& mRect = Rectf(mTrack->minx, mTrack->miny, mTrack->maxx, mTrack->maxy);
                gl::drawStrokedRect(mRect);
            }
        }    
    }
}

CINDER_APP_BASIC( SimpleBlobDetectionApp, RendererGl )
