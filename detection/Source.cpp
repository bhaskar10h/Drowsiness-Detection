#include<iostream>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>


using namespace dlib;
using namespace std;
using namespace cv;


image_window win;
shape_predictor sp;
std::vector<cv::Point> righteye;
std::vector<cv::Point> lefteye;
char c;
cv::Point p;

double compute_EAR(std::vector<cv::Point> vec)
{
    double a = cv::norm(cv::Mat(vec[1]), cv::Mat(vec[5]));
    double b = cv::norm(cv::Mat(vec[2]), cv::Mat(vec[4]));
    double c = cv::norm(cv::Mat(vec[0]), cv::Mat(vec[3]));
    double ear = (a + b) / (2.0 * c);
    return ear;
}

int main()
{
    try {
        cv::VideoCapture cap(0);

        if (!cap.isOpened()) {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }
        cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

        frontal_face_detector detector = get_frontal_face_detector();
        deserialize("/path/to/model/shape_predictor_68_face_landmarks.dat") >> sp;
        
        // Create a pose predictor for head pose estimation
        anypoint_model model = pose_predictor<anypoint_model>(sp);

        bool isBlinking = false;
        int blinkCount = 0;

        cv::VideoWriter writer("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(640, 480));

        while (!win.is_closed()) {
            cv::Mat temp;
            if (!cap.read(temp)) {
                break;
            }

            cv_image<bgr_pixel> cimg(temp);
            full_object_detection shape;

            std::vector<rectangle> faces = detector(cimg);
            cout << "Number of faces detected: " << faces.size() << endl;

            win.clear_overlay();
            win.set_image(cimg);

            for (size_t i = 0; i < faces.size(); ++i) {
                shape = sp(cimg, faces[i]);

                // Head Pose Estimation
                full_object_detection pose = model(cimg, faces[i]);

                for (int b = 36; b < 42; ++b) {
                    p.x = shape.part(b).x();
                    p.y = shape.part(b).y();
                    lefteye.push_back(p);
                }
                for (int b = 42; b < 48; ++b) {
                    p.x = shape.part(b).x();
                    p.y = shape.part(b).y();
                    righteye.push_back(p);
                }

                double right_ear = compute_EAR(righteye);
                double left_ear = compute_EAR(lefteye);

                if (left_ear < 0.2 && right_ear < 0.2) {
                    if (!isBlinking) {
                        blinkCount++;
                        isBlinking = true;
                    }
                } else {
                    isBlinking = false;
                }

                // Drawing overlay based on EAR
                if ((right_ear + left_ear) / 2 < 0.2)
                    win.add_overlay(dlib::image_window::overlay_rect(faces[i], rgb_pixel(255, 255, 255), "Sleeping"));
                else
                    win.add_overlay(dlib::image_window::overlay_rect(faces[i], rgb_pixel(255, 255, 255), "Not sleeping"));

                righteye.clear();
                lefteye.clear();

                win.add_overlay(render_face_detections(shape));
            }

            // Record the frame
            writer.write(temp);

            c = (char)waitKey(30);
            if (c == 27)
                break;
        }

        writer.release();  // Release the video writer

    } catch (serialization_error& e) {
        cout << "Check the path to dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl
            << e.what() << endl;
    } catch (exception& e) {
        cout << e.what() << endl;
    }

    return 0;
}
