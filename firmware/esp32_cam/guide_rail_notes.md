# ESP32-CAM Guide Rail Alignment Guide

To inspect elevator guide rails:
1. Mount the ESP32-CAM module to the elevator car frame or counterweight guide assembly, facing the machined blade of the guide rail.
2. Distance to rail blade: 100mm - 250mm.
3. Lighting: Built-in high power flash LED (`GET /flash?on=1`) or ambient shaft luminaire.
4. Optical focus: Adjust OV2640 manual focus ring until the guide rail surface machining grooves and joint fishplates are in sharp focus.
5. Ingestion: ESP32-S3 continuously fetches frames over local WiFi/AP at `/capture` and extracts wear lines and joint steps.
