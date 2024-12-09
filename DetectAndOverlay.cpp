#include <opencv2/highgui/highgui.hpp>    // Thư viện OpenCV hỗ trợ giao diện người dùng (UI) như cửa sổ, thao tác với hình ảnh.
#include <opencv2/imgproc/imgproc.hpp>    // Thư viện OpenCV cung cấp các hàm xử lý ảnh (như chuyển đổi màu, lọc ảnh).
#include <opencv2/objdetect/objdetect.hpp> // Thư viện OpenCV hỗ trợ phát hiện đối tượng (bao gồm khuôn mặt).
#include <iostream>                        // Thư viện chuẩn C++ cho việc nhập xuất dữ liệu, in thông báo lỗi.
#include <stdexcept>                       // Thư viện cho việc xử lý ngoại lệ (throw/catch lỗi).

using namespace cv;  // Sử dụng không gian tên OpenCV để gọi các hàm và lớp mà không cần viết 'cv::' trước mỗi tên.
using namespace std; // Sử dụng không gian tên chuẩn của C++ để gọi các đối tượng như 'string', 'cout', 'cerr', v.v.

// Interface for FaceDetector - Giao diện cho lớp FaceDetector, yêu cầu các lớp con phải triển khai phương thức detectAndApplyMask.
class IFaceDetector {
public:
    // Phương thức ảo để phát hiện và áp dụng mặt nạ lên khuôn mặt trong ảnh
    virtual void detectAndApplyMask(Mat& frame) = 0;
    virtual ~IFaceDetector() = default;  // Destructor mặc định, có thể hủy các tài nguyên khi không cần nữa
};

// FaceDetector class - Lớp FaceDetector kế thừa từ IFaceDetector, thực hiện việc phát hiện khuôn mặt và áp dụng mặt nạ lên khuôn mặt.
class FaceDetector : public IFaceDetector {
private:
    CascadeClassifier faceCascade;  // Đối tượng CascadeClassifier dùng để phát hiện khuôn mặt trong ảnh (dựa trên phương pháp haar-cascade).
    Mat faceMask;                   // Mặt nạ sẽ được áp dụng lên khuôn mặt (một ảnh PNG hoặc ảnh mặt nạ).
    float scalingFactor;            // Hệ số thu nhỏ ảnh để tăng tốc quá trình phát hiện khuôn mặt.

    // Hàm để tải file cascade (xml chứa các mô hình học máy phát hiện khuôn mặt)
    void loadCascade(const string& cascadePath) {
        if (!faceCascade.load(cascadePath)) {
            throw runtime_error("Error loading cascade file.");  // Thông báo lỗi nếu không thể tải cascade từ file
        }
    }

    // Hàm tải ảnh mặt nạ từ đường dẫn file.
    void loadFaceMask(const string& maskPath) {
        faceMask = imread(maskPath);  // Đọc mặt nạ từ file
        if (!faceMask.data) {
            throw runtime_error("Error loading mask image."); // Thông báo lỗi nếu không thể tải mặt nạ
        }
    }

public:
    // Constructor nhận đường dẫn đến cascade và mặt nạ cùng với hệ số thu nhỏ mặc định (0.75).
    FaceDetector(const string& cascadePath, const string& maskPath, float scale = 0.75)
        : scalingFactor(scale) {
        loadCascade(cascadePath);    // Gọi hàm tải cascade từ đường dẫn
        loadFaceMask(maskPath);      // Gọi hàm tải mặt nạ từ đường dẫn
    }

    // Phương thức thực hiện phát hiện khuôn mặt và áp dụng mặt nạ lên khuôn mặt trong ảnh
    void detectAndApplyMask(Mat& frame) override {
        Mat frameGray, frameROI, faceMaskSmall;
        Mat grayMaskSmall, grayMaskSmallThresh, grayMaskSmallThreshInv;
        Mat maskedFace, maskedFrame;

        // Resize ảnh để giảm kích thước và tăng tốc độ xử lý (sử dụng hệ số scalingFactor)
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

        // Chuyển ảnh gốc từ màu sắc sang ảnh grayscale để dễ dàng xử lý
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        equalizeHist(frameGray, frameGray);  // Cân bằng độ sáng trong ảnh grayscale

        // Phát hiện khuôn mặt trong ảnh grayscale
        vector<Rect> faces;
        faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0 | 2, Size(30, 30));

        // Duyệt qua tất cả các khuôn mặt đã được phát hiện
        for (const auto& face : faces) {
            // Xác định vị trí và kích thước khuôn mặt (có thể mở rộng thêm chút ở các cạnh để bao phủ hết mặt)
            int x = face.x - int(0.1 * face.width);  // Xác định vị trí x của khuôn mặt
            int y = face.y - int(0.0 * face.height); // Xác định vị trí y của khuôn mặt
            int w = int(1.1 * face.width);           // Chiều rộng khuôn mặt
            int h = int(1.3 * face.height);          // Chiều cao khuôn mặt

            // Kiểm tra xem khuôn mặt có nằm trong phạm vi của ảnh không
            if (x > 0 && y > 0 && x + w < frame.cols && y + h < frame.rows) {
                // Lấy vùng ảnh chứa khuôn mặt
                frameROI = frame(Rect(x, y, w, h));
                
                // Resize mặt nạ cho phù hợp với kích thước khuôn mặt
                resize(faceMask, faceMaskSmall, Size(w, h));
                
                // Chuyển mặt nạ thành ảnh grayscale
                cvtColor(faceMaskSmall, grayMaskSmall, COLOR_BGR2GRAY);

                // Ngưỡng hóa mặt nạ thành nhị phân (sáng/tối)
                threshold(grayMaskSmall, grayMaskSmallThresh, 230, 255, THRESH_BINARY_INV);
                bitwise_not(grayMaskSmallThresh, grayMaskSmallThreshInv);

                // Áp dụng mặt nạ vào khuôn mặt
                bitwise_and(faceMaskSmall, faceMaskSmall, maskedFace, grayMaskSmallThresh);
                bitwise_and(frameROI, frameROI, maskedFrame, grayMaskSmallThreshInv);

                // Ghép mặt nạ vào khuôn mặt trong ảnh
                add(maskedFace, maskedFrame, frame(Rect(x, y, w, h)));
            }
        }
    }
};

// Lớp ứng dụng sử dụng FaceDetector để áp dụng mặt nạ lên khuôn mặt trong video (webcam)
class FaceMaskApplication {
private:
    unique_ptr<IFaceDetector> detector;  // Đối tượng detector (sử dụng FaceDetector hoặc lớp khác kế thừa IFaceDetector)
    VideoCapture cap;                    // Đối tượng để xử lý video từ webcam

public:
    // Constructor nhận một đối tượng detector để phát hiện và áp dụng mặt nạ, và mở webcam
    FaceMaskApplication(unique_ptr<IFaceDetector> detector) : detector(move(detector)) {
        if (!cap.open(0)) {
            throw runtime_error("Error opening webcam.");  // Thông báo lỗi nếu không mở được webcam
        }
        namedWindow("Frame");  // Tạo cửa sổ để hiển thị các frame video
    }

    // Phương thức chạy ứng dụng, xử lý từng frame từ webcam
    void run() {
        Mat frame;
        while (true) {
            cap >> frame;  // Lấy frame tiếp theo từ webcam
            if (frame.empty()) {
                cerr << "Error capturing frame. Exiting!" << endl;  // Thông báo lỗi nếu không lấy được frame
                break;
            }

            try {
                // Phát hiện khuôn mặt và áp dụng mặt nạ lên khuôn mặt
                detector->detectAndApplyMask(frame);
            }
            catch (const exception& e) {
                cerr << e.what() << endl;  // In thông báo lỗi nếu có lỗi trong quá trình phát hiện và áp dụng mặt nạ
                break;
            }

            imshow("Frame", frame);  // Hiển thị frame sau khi đã áp dụng mặt nạ
            if (waitKey(30) == 27) {  // Kiểm tra phím nhấn 'Esc' để thoát khỏi vòng lặp
                break;
            }
        }
        cap.release();  // Giải phóng tài nguyên webcam khi không cần nữa
        destroyAllWindows();  // Đóng tất cả các cửa sổ hiển thị ảnh
    }
};

// NoseDetector class - Lớp phát hiện mũi, kế thừa từ IFaceDetector
class NoseDetector : public IFaceDetector {
private:
    CascadeClassifier faceCascade; // CascadeClassifier để phát hiện khuôn mặt
    CascadeClassifier noseCascade; // CascadeClassifier để phát hiện mũi trong khuôn mặt
    Mat noseMask;                  // Mặt nạ để áp dụng lên mũi
    float scalingFactor;           // Hệ số thu nhỏ cho ảnh, dùng để tăng tốc độ phát hiện

    // Hàm tải cascade (một phương pháp học máy) từ tệp xml
    void loadCascade(const string& cascadePath, CascadeClassifier& cascade) {
        if (!cascade.load(cascadePath)) {
            throw runtime_error("Error loading cascade file: " + cascadePath); // Lỗi khi không tải được cascade
        }
    }

    // Hàm tải ảnh mặt nạ (mũi)
    void loadNoseMask(const string& maskPath) {
        noseMask = imread(maskPath); // Đọc ảnh mặt nạ từ tệp
        if (!noseMask.data) {
            throw runtime_error("Error loading mask image."); // Lỗi khi không tải được ảnh mặt nạ
        }
    }

public:
    // Constructor nhận các đường dẫn đến cascade khuôn mặt, cascade mũi, và mặt nạ mũi
    NoseDetector(const string& faceCascadePath, const string& noseCascadePath, const string& maskPath, float scale = 0.75)
        : scalingFactor(scale) {
        loadCascade(faceCascadePath, faceCascade); // Tải cascade khuôn mặt
        loadCascade(noseCascadePath, noseCascade); // Tải cascade mũi
        loadNoseMask(maskPath);                    // Tải mặt nạ mũi
    }

    // Phương thức thực hiện phát hiện mũi và áp dụng mặt nạ mũi lên ảnh
    void detectAndApplyMask(Mat& frame) override {
        Mat frameGray, frameROI, noseMaskSmall;     // Các biến ảnh con và mặt nạ mũi nhỏ
        Mat grayMaskSmall, grayMaskSmallThresh, grayMaskSmallThreshInv; // Biến để xử lý mặt nạ mũi
        Mat maskedNose, maskedFrame;  // Biến để lưu kết quả mặt nạ mũi và kết quả cuối cùng

        // Resize ảnh để giảm kích thước và tăng tốc độ xử lý
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

        // Chuyển ảnh gốc sang ảnh grayscale
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        equalizeHist(frameGray, frameGray); // Cân bằng độ sáng trong ảnh

        // Phát hiện khuôn mặt trong ảnh
        vector<Rect> faces;
        faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        // Duyệt qua tất cả các khuôn mặt phát hiện được
        for (const auto& face : faces) {
            Mat faceROI = frameGray(face); // Lấy ROI (vùng quan tâm) khuôn mặt
            vector<Rect> noses;           // Mảng chứa các vị trí mũi phát hiện được trong khuôn mặt

            // Phát hiện mũi trong khuôn mặt
            noseCascade.detectMultiScale(faceROI, noses, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

            // Duyệt qua tất cả các mũi phát hiện được trong khuôn mặt
            for (const auto& nose : noses) {
                // Tính toán vị trí và kích thước của mặt nạ mũi
                int x = face.x + nose.x - int(0.1 * nose.width);
                int y = face.y + nose.y - int(0.3 * nose.height);
                int w = int(1.3 * nose.width);
                int h = int(1.7 * nose.height);

                // Kiểm tra nếu mặt nạ mũi có nằm trong phạm vi của ảnh không
                if (x > 0 && y > 0 && x + w < frame.cols && y + h < frame.rows) {
                    frameROI = frame(Rect(x, y, w, h)); // Lấy ROI mũi từ ảnh gốc
                    resize(noseMask, noseMaskSmall, Size(w, h)); // Resize mặt nạ mũi cho phù hợp với khuôn mặt
                    cvtColor(noseMaskSmall, grayMaskSmall, COLOR_BGR2GRAY); // Chuyển mặt nạ mũi thành grayscale

                    // Ngưỡng hóa mặt nạ để tạo ảnh nhị phân (sáng/tối)
                    threshold(grayMaskSmall, grayMaskSmallThresh, 250, 255, THRESH_BINARY_INV);
                    bitwise_not(grayMaskSmallThresh, grayMaskSmallThreshInv); // Lật ảnh nhị phân

                    // Áp dụng mặt nạ mũi lên ảnh
                    bitwise_and(noseMaskSmall, noseMaskSmall, maskedNose, grayMaskSmallThresh);
                    bitwise_and(frameROI, frameROI, maskedFrame, grayMaskSmallThreshInv);

                    // Ghép mặt nạ mũi vào ảnh
                    add(maskedNose, maskedFrame, frame(Rect(x, y, w, h)));
                }
            }
        }
    }
};

// EyeDetector class - Lớp phát hiện mắt, kế thừa từ IFaceDetector
class EyeDetector : public IFaceDetector {
private:
    CascadeClassifier faceCascade; // CascadeClassifier để phát hiện khuôn mặt
    CascadeClassifier eyeCascade;  // CascadeClassifier để phát hiện mắt trong khuôn mặt
    Mat eyeMask;                   // Mặt nạ để áp dụng lên mắt
    float scalingFactor;           // Hệ số thu nhỏ cho ảnh, dùng để tăng tốc độ phát hiện

    // Hàm tải cascade (một phương pháp học máy) từ tệp xml
    void loadCascade(const string& cascadePath, CascadeClassifier& cascade) {
        if (!cascade.load(cascadePath)) {
            throw runtime_error("Error loading cascade file: " + cascadePath); // Lỗi khi không tải được cascade
        }
    }

    // Hàm tải ảnh mặt nạ (mắt)
    void loadEyeMask(const string& maskPath) {
        eyeMask = imread(maskPath); // Đọc ảnh mặt nạ từ tệp
        if (!eyeMask.data) {
            throw runtime_error("Error loading mask image."); // Lỗi khi không tải được ảnh mặt nạ
        }
    }

public:
    // Constructor nhận các đường dẫn đến cascade khuôn mặt, cascade mắt và mặt nạ mắt
    EyeDetector(const string& faceCascadePath, const string& eyeCascadePath, const string& maskPath, float scale = 0.75)
        : scalingFactor(scale) {
        loadCascade(faceCascadePath, faceCascade); // Tải cascade khuôn mặt
        loadCascade(eyeCascadePath, eyeCascade);    // Tải cascade mắt
        loadEyeMask(maskPath);                       // Tải mặt nạ mắt
    }

    // Phương thức thực hiện phát hiện mắt và áp dụng mặt nạ mắt lên ảnh
    void detectAndApplyMask(Mat& frame) override {
        Mat frameGray, frameROI, eyeMaskSmall;
        Mat grayMaskSmall, grayMaskSmallThresh, grayMaskSmallThreshInv;
        Mat maskedEye, maskedFrame;

        // Resize ảnh để giảm kích thước và tăng tốc độ xử lý
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

        // Chuyển ảnh gốc sang ảnh grayscale
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        equalizeHist(frameGray, frameGray); // Cân bằng độ sáng trong ảnh

        // Phát hiện khuôn mặt trong ảnh
        vector<Rect> faces;
        faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        vector<Point> centers;  // Mảng lưu các điểm trung tâm của mắt

        // Duyệt qua tất cả các khuôn mặt phát hiện được
        for (const auto& face : faces) {
            Mat faceROI = frameGray(face); // Lấy ROI (vùng quan tâm) khuôn mặt
            vector<Rect> eyes;            // Mảng chứa các vị trí mắt phát hiện được trong khuôn mặt

            // Phát hiện mắt trong khuôn mặt
            eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

            // Tính toán vị trí trung tâm của các mắt phát hiện
            for (const auto& eye : eyes) {
                Point center(face.x + eye.x + int(eye.width * 0.5), face.y + eye.y + int(eye.height * 0.5));
                centers.push_back(center);
            }
        }

        // Áp dụng mặt nạ kính mát nếu có phát hiện 2 mắt
        if (centers.size() == 2) {
            Point leftPoint, rightPoint;

            // Xác định mắt trái và mắt phải
            if (centers[0].x < centers[1].x) {
                leftPoint = centers[0];
                rightPoint = centers[1];
            } else {
                leftPoint = centers[1];
                rightPoint = centers[0];
            }

            // Tính toán kích thước và vị trí của mặt nạ kính mát
            int w = 2.3 * (rightPoint.x - leftPoint.x);
            int h = int(0.4 * w);
            int x = leftPoint.x - int(0.25 * w);
            int y = leftPoint.y - int(0.5 * h);

            // Kiểm tra nếu mặt nạ kính mát có nằm trong phạm vi của ảnh không
            if (x > 0 && y > 0 && x + w < frame.cols && y + h < frame.rows) {
                frameROI = frame(Rect(x, y, w, h)); // Lấy ROI kính mát từ ảnh gốc
                resize(eyeMask, eyeMaskSmall, Size(w, h)); // Resize mặt nạ kính mát cho phù hợp với khuôn mặt
                cvtColor(eyeMaskSmall, grayMaskSmall, COLOR_BGR2GRAY); // Chuyển mặt nạ kính mát thành grayscale

                // Ngưỡng hóa mặt nạ để tạo ảnh nhị phân (sáng/tối)
                threshold(grayMaskSmall, grayMaskSmallThresh, 245, 255, THRESH_BINARY_INV);
                bitwise_not(grayMaskSmallThresh, grayMaskSmallThreshInv); // Lật ảnh nhị phân

                // Áp dụng mặt nạ kính mát lên ảnh
                bitwise_and(eyeMaskSmall, eyeMaskSmall, maskedEye, grayMaskSmallThresh);
                bitwise_and(frameROI, frameROI, maskedFrame, grayMaskSmallThreshInv);

                // Ghép mặt nạ kính mát vào ảnh
                add(maskedEye, maskedFrame, frame(Rect(x, y, w, h)));
            }
        }
    }
};

// EarDetector class
class EarDetector : public IFaceDetector {
private:
    CascadeClassifier leftEarCascade;  // Cascade để phát hiện tai trái
    CascadeClassifier rightEarCascade; // Cascade để phát hiện tai phải
    float scalingFactor;              // Hệ số thu nhỏ ảnh để tăng tốc độ phát hiện

    // Hàm tải cascade từ tệp
    void loadCascade(const string& cascadePath, CascadeClassifier& cascade) {
        if (!cascade.load(cascadePath)) {
            throw runtime_error("Error loading cascade file: " + cascadePath);
        }
    }

public:
    // Constructor nhận các đường dẫn đến cascade tai trái và tai phải, cùng với hệ số thu nhỏ
    EarDetector(const string& leftEarCascadePath, const string& rightEarCascadePath, float scale = 0.75)
        : scalingFactor(scale) {
        loadCascade(leftEarCascadePath, leftEarCascade);
        loadCascade(rightEarCascadePath, rightEarCascade);
    }

    // Hàm phát hiện và vẽ mặt nạ tai lên ảnh
    void detectAndApplyMask(Mat& frame) override {
        Mat frameGray;
        vector<Rect> leftEars, rightEars;

        // Resize ảnh để giảm kích thước và tăng tốc độ xử lý
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

        // Chuyển ảnh gốc sang grayscale
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        equalizeHist(frameGray, frameGray);

        // Phát hiện tai trái
        leftEarCascade.detectMultiScale(frameGray, leftEars, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        // Phát hiện tai phải
        rightEarCascade.detectMultiScale(frameGray, rightEars, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        // Vẽ hình chữ nhật xung quanh tai trái phát hiện được
        for (const auto& leftEar : leftEars) {
            Rect leftEarRect(leftEar.x, leftEar.y, leftEar.width, leftEar.height);
            rectangle(frame, leftEarRect, Scalar(0, 255, 0), 4);
        }

        // Vẽ hình chữ nhật xung quanh tai phải phát hiện được
        for (const auto& rightEar : rightEars) {
            Rect rightEarRect(rightEar.x, rightEar.y, rightEar.width, rightEar.height);
            rectangle(frame, rightEarRect, Scalar(0, 255, 0), 4);
        }
    }
};

// MoustacheDetector class
class MoustacheDetector : public IFaceDetector {
private:
    CascadeClassifier faceCascade;  // Cascade để phát hiện khuôn mặt
    CascadeClassifier mouthCascade; // Cascade để phát hiện miệng
    Mat moustacheMask;              // Mặt nạ để gắn lên miệng
    float scalingFactor;            // Hệ số thu nhỏ ảnh

    // Hàm tải cascade từ tệp
    void loadCascade(const string& cascadePath, CascadeClassifier& cascade) {
        if (!cascade.load(cascadePath)) {
            throw runtime_error("Error loading cascade file: " + cascadePath);
        }
    }

    // Hàm tải mặt nạ từ tệp
    void loadMask(const string& maskPath, Mat& mask) {
        mask = imread(maskPath);
        if (!mask.data) {
            throw runtime_error("Error loading moustache mask image: " + maskPath);
        }
    }

public:
    MoustacheDetector(const string& faceCascadePath, const string& mouthCascadePath, const string& maskPath, float scale = 0.75)
        : scalingFactor(scale) {
        loadCascade(faceCascadePath, faceCascade);
        loadCascade(mouthCascadePath, mouthCascade);
        loadMask(maskPath, moustacheMask);
    }

    // Hàm phát hiện và vẽ mặt nạ ria mép lên ảnh
    void detectAndApplyMask(Mat& frame) override {
        Mat frameGray;
        vector<Rect> faces;

        // Resize ảnh để giảm kích thước và tăng tốc độ xử lý
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

        // Chuyển ảnh gốc sang grayscale
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);

        // Phát hiện khuôn mặt
        faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        // Xử lý từng khuôn mặt phát hiện được
        for (const auto& face : faces) {
            Mat faceROI = frameGray(face);
            vector<Rect> mouths;

            // Phát hiện miệng trong khuôn mặt
            mouthCascade.detectMultiScale(faceROI, mouths, 1.1, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

            for (const auto& mouth : mouths) {
                int w = 1.8 * mouth.width;
                int h = mouth.height;
                int x = face.x + mouth.x - 0.2 * w;
                int y = face.y + mouth.y + 0.65 * h;

                // Tạo ROI cho việc đặt ria mép
                Mat frameROI = frame(Rect(x, y, w, h));

                // Resize mặt nạ ria mép để vừa với ROI
                Mat moustacheMaskResized;
                resize(moustacheMask, moustacheMaskResized, Size(w, h));

                // Tạo mask grayscale và áp dụng threshold
                Mat grayMask, grayMaskThresh, grayMaskThreshInv;
                cvtColor(moustacheMaskResized, grayMask, COLOR_BGR2GRAY);
                threshold(grayMask, grayMaskThresh, 245, 255, THRESH_BINARY_INV);
                bitwise_not(grayMaskThresh, grayMaskThreshInv);

                // Áp dụng mask ria mép và frame
                Mat maskedMoustache, maskedFrame;
                bitwise_and(moustacheMaskResized, moustacheMaskResized, maskedMoustache, grayMaskThresh);
                bitwise_and(frameROI, frameROI, maskedFrame, grayMaskThreshInv);

                // Kết hợp các ảnh đã mask
                add(maskedMoustache, maskedFrame, frame(Rect(x, y, w, h)));
            }
        }
    }
};

// Main function
int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <mode> <cascade_file_1> <cascade_file_2|mask_image>" << endl;
        cerr << "Modes: face | nose | eye | ear | moustache" << endl;
        return -1;
    }

    string mode = argv[1];

    try {
        unique_ptr<IFaceDetector> detector;
        if (mode == "face") {
            detector = make_unique<FaceDetector>(argv[2], argv[3]);
        }
        else if (mode == "nose") {
            if (argc < 5) {
                cerr << "Usage for nose mode: " << argv[0] << " nose <face_cascade> <nose_cascade> <mask_image>" << endl;
                return -1;
            }
            detector = make_unique<NoseDetector>(argv[2], argv[3], argv[4]);
        }
        else if (mode == "eye") {
            if (argc < 5) {
                cerr << "Usage for eye mode: " << argv[0] << " eye <face_cascade> <eye_cascade> <mask_image>" << endl;
                return -1;
            }
            detector = make_unique<EyeDetector>(argv[2], argv[3], argv[4]);
        }
        else if (mode == "ear") {
            detector = make_unique<EarDetector>(argv[2], argv[3]);
        }
        else if (mode == "moustache") {
            if (argc < 5) {
                cerr << "Usage for moustache mode: " << argv[0] << " moustache <face_cascade> <mouth_cascade> <mask_image>" << endl;
                return -1;
            }
            detector = make_unique<MoustacheDetector>(argv[2], argv[3], argv[4]);
        }
        else {
            cerr << "Invalid mode: " << mode << endl;
            return -1;
        }

        FaceMaskApplication app(move(detector));
        app.run();
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}
