#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap;
    std::string choice;
    std::cout << "Seleccione fuente:\n1 - Video\n2 - Cámara\nOpción: ";
    std::cin >> choice;

    if (choice == "1") {
        std::string filename;
        std::cout << "Ingrese ruta del video: ";
        std::cin >> filename;
        cap.open(filename);
    } else {
        cap.open(0);
    }

    if (!cap.isOpened()) {
        std::cerr << "Error al abrir la fuente" << std::endl;
        return -1;
    }

    cv::Mat frame, displayOriginal, resultCPU;
    cv::cuda::GpuMat d_frame, d_gray, d_equalized, d_blur, d_morph, d_edges;

    // Crear filtros GPU UNA sola vez
    auto gauss = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 1.5);
    auto morph = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1,
                    cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    auto canny = cv::cuda::createCannyEdgeDetector(50, 150);

    bool isVideo = (choice == "1");
    double fps = cap.get(cv::CAP_PROP_FPS);
    int delay = (fps > 0 && isVideo) ? int(1000 / fps) : 1;

    int frameCount = 0;
    double t0 = (double)cv::getTickCount();

    while (true) {
        if (!cap.read(frame)) break;

        // Reducir resolución para GPU más rápida (opcional)
        cv::resize(frame, displayOriginal, cv::Size(), 0.5, 0.5);

        // ------------------------
        // GPU Pipeline
        // ------------------------
        double t_start = (double)cv::getTickCount();

        d_frame.upload(displayOriginal);               // CPU → GPU
        cv::cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY); // GPU
        cv::cuda::equalizeHist(d_gray, d_equalized);             // GPU
        gauss->apply(d_equalized, d_blur);                        // GPU
        morph->apply(d_blur, d_morph);                            // GPU
        canny->detect(d_morph, d_edges);                          // GPU
        d_edges.download(resultCPU);                              // GPU → CPU

        double t_end = (double)cv::getTickCount();
        double time_ms = (t_end - t_start) * 1000.0 / cv::getTickFrequency();
        frameCount++;

        double elapsedSec = (t_end - t0) / cv::getTickFrequency();
        double realFPS = frameCount / elapsedSec;

        // Mostrar FPS y tiempo GPU por frame
        cv::putText(resultCPU,
            "GPU Time: " + std::to_string(int(time_ms)) + " ms | FPS: " + std::to_string(int(realFPS)),
            cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(255),
            2
        );

        // Mostrar ventanas
        cv::imshow("Original", displayOriginal);
        cv::imshow("GPU Pipeline Result", resultCPU);

        if (cv::waitKey(delay) == 27) break; // ESC
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
