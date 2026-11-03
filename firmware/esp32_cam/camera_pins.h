#pragma once

// Select Camera Model (Uncomment only one)
#define CAMERA_MODEL_AI_THINKER // Has PSRAM
//#define CAMERA_MODEL_WROVER_KIT
//#define CAMERA_MODEL_ESP_EYE
//#define CAMERA_MODEL_M5STACK_PSRAM

#if defined(CAMERA_MODEL_AI_THINKER)
  #define PWDN_GPIO_NUM     32
  #define RESET_GPIO_NUM    -1
  #define XCLK_GPIO_NUM      0
  #define SIOD_GPIO_NUM     26
  #define SIOC_GPIO_NUM     27

  #define Y9_GPIO_NUM       35
  #define Y8_GPIO_NUM       34
  #define Y7_GPIO_NUM       39
  #define Y6_GPIO_NUM       36
  #define Y5_GPIO_NUM       21
  #define Y4_GPIO_NUM       19
  #define Y3_GPIO_NUM       18
  #define Y2_GPIO_NUM        5

  #define VSYNC_GPIO_NUM    25
  #define HREF_GPIO_NUM     23
  #define PCLK_GPIO_NUM     22

  // Onboard Flash / LED Pins
  #define FLASH_LED_PIN      4
  #define ONBOARD_LED_PIN   33

#elif defined(CAMERA_MODEL_WROVER_KIT)
  #define PWDN_GPIO_NUM    -1
  #define RESET_GPIO_NUM   -1
  #define XCLK_GPIO_NUM    21
  #define SIOD_GPIO_NUM    26
  #define SIOC_GPIO_NUM    27

  #define Y9_GPIO_NUM      35
  #define Y8_GPIO_NUM      34
  #define Y7_GPIO_NUM      39
  #define Y6_GPIO_NUM      36
  #define Y5_GPIO_NUM      19
  #define Y4_GPIO_NUM      18
  #define Y3_GPIO_NUM       5
  #define Y2_GPIO_NUM       4
  #define VSYNC_GPIO_NUM   25
  #define HREF_GPIO_NUM    23
  #define PCLK_GPIO_NUM    22

  #define FLASH_LED_PIN    -1
  #define ONBOARD_LED_PIN   2

#elif defined(CAMERA_MODEL_M5STACK_PSRAM)
  #define PWDN_GPIO_NUM    -1
  #define RESET_GPIO_NUM   15
  #define XCLK_GPIO_NUM    27
  #define SIOD_GPIO_NUM    25
  #define SIOC_GPIO_NUM    23

  #define Y9_GPIO_NUM      19
  #define Y8_GPIO_NUM      36
  #define Y7_GPIO_NUM      18
  #define Y6_GPIO_NUM      39
  #define Y5_GPIO_NUM       5
  #define Y4_GPIO_NUM      34
  #define Y3_GPIO_NUM      35
  #define Y2_GPIO_NUM      32
  #define VSYNC_GPIO_NUM   22
  #define HREF_GPIO_NUM    26
  #define PCLK_GPIO_NUM    21

  #define FLASH_LED_PIN    -1
  #define ONBOARD_LED_PIN  -1

#else
  #error "Camera model not selected! Please define a model in camera_pins.h"
#endif
