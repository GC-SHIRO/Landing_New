#include <gazebo/gazebo_client.hh>
#include <gazebo/msgs/image_stamped.pb.h>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/transport/transport.hh>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{
std::mutex frameMutex;
std::vector<unsigned char> latestFrame;
unsigned int frameWidth = 0;
unsigned int frameHeight = 0;
std::atomic<unsigned long> messageCount{0};

void OnImage(const ConstImageStampedPtr &_message)
{
  const auto &image = _message->image();
  const std::size_t expected =
      static_cast<std::size_t>(image.width()) * image.height() * 3;
  if (image.data().size() != expected)
    return;

  std::lock_guard<std::mutex> guard(frameMutex);
  frameWidth = image.width();
  frameHeight = image.height();
  latestFrame.assign(image.data().begin(), image.data().end());
  ++messageCount;
}
}  // namespace

int main(int argc, char **argv)
{
  if (argc != 5)
  {
    std::cerr << "usage: " << argv[0]
              << " TOPIC OUTPUT_MP4 DURATION_SECONDS FPS\n";
    return 2;
  }

  const std::string topic = argv[1];
  const std::string output = argv[2];
  const double duration = std::stod(argv[3]);
  const double fps = std::stod(argv[4]);
  if (duration <= 0.0 || fps <= 0.0)
    return 2;

  gazebo::client::setup(argc, argv);
  gazebo::transport::NodePtr node(new gazebo::transport::Node());
  node->Init();
  auto subscriber = node->Subscribe(topic, &OnImage);

  const auto firstFrameDeadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(45);
  while (messageCount.load() == 0 &&
         std::chrono::steady_clock::now() < firstFrameDeadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

  if (messageCount.load() == 0)
  {
    std::cerr << "no Gazebo image received from " << topic << "\n";
    gazebo::client::shutdown();
    return 3;
  }

  unsigned int width;
  unsigned int height;
  {
    std::lock_guard<std::mutex> guard(frameMutex);
    width = frameWidth;
    height = frameHeight;
  }

  const std::string partial = output + ".part.mp4";
  std::ostringstream command;
  command << "ffmpeg -nostdin -y -loglevel error -f rawvideo -pixel_format rgb24"
          << " -video_size " << width << "x" << height
          << " -framerate " << fps << " -i - -an -c:v libx264"
          << " -preset veryfast -crf 20 -pix_fmt yuv420p"
          << " -movflags +faststart " << partial;

  FILE *encoder = popen(command.str().c_str(), "w");
  if (!encoder)
  {
    gazebo::client::shutdown();
    return 4;
  }

  const int targetFrames = static_cast<int>(duration * fps + 0.5);
  const auto interval = std::chrono::duration<double>(1.0 / fps);
  const auto started = std::chrono::steady_clock::now();
  for (int index = 0; index < targetFrames; ++index)
  {
    std::this_thread::sleep_until(started + interval * index);
    std::vector<unsigned char> frame;
    {
      std::lock_guard<std::mutex> guard(frameMutex);
      frame = latestFrame;
    }
    if (std::fwrite(frame.data(), 1, frame.size(), encoder) != frame.size())
    {
      pclose(encoder);
      gazebo::client::shutdown();
      return 5;
    }
  }

  const int encoderStatus = pclose(encoder);
  gazebo::client::shutdown();
  if (encoderStatus != 0)
    return 6;
  if (std::rename(partial.c_str(), output.c_str()) != 0)
    return 7;

  const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  std::cout << "VIDEO_DONE output=" << output
            << " frames=" << targetFrames
            << " messages=" << messageCount.load()
            << " elapsed=" << elapsed
            << " size=" << width << "x" << height << "\n";
  return 0;
}
