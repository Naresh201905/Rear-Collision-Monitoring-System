
// Include necessary libraries
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// Initialize LCD (address, columns, rows)
// Default I2C address is 0x27, adjust if your LCD uses a different address
LiquidCrystal_I2C lcd(0x27, 16, 2);

// Pin definitions
const int trigPin = 9;    // Trigger pin of ultrasonic sensor
const int echoPin = 10;   // Echo pin of ultrasonic sensor
const int buzzerPin = 8;  // Buzzer pin

// Constants
const int DISTANCE_THRESHOLD = 50; // Distance threshold in cm for warning (as per paper)
const int MEASUREMENT_INTERVAL = 100; // Time between measurements in ms

// Variables
long duration;
int distance;
String alertMessage = "";
unsigned long lastDisplayUpdate = 0;
const unsigned long DISPLAY_UPDATE_INTERVAL = 300; // Update display every 300ms to prevent flicker

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Configure pin modes
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(buzzerPin, OUTPUT);
  
  // Initialize the LCD
  lcd.init();
  lcd.backlight();
  
  // Display startup message
  lcd.setCursor(0, 0);
  lcd.print("Collision Avoid");
  lcd.setCursor(0, 1);
  lcd.print("System Ready");
  delay(2000);
  lcd.clear();
  
  // Initial message to serial
  Serial.println("Rear Collision Avoidance System - Ultrasonic Sensor Ready");
}

void loop() {
  // Clear the trigger pin
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  
  // Set the trigger pin high for 10 microseconds
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // Read the echo pin - returns sound wave travel time in microseconds
  duration = pulseIn(echoPin, HIGH);
  
  // Calculate distance: speed of sound is 343 m/s or 0.0343 cm/μs
  // Time is for round trip so divide by 2
  distance = duration * 0.0343 / 2;
  
  // Send distance data to serial (for Python code to read)
  Serial.print("Distance: ");
  Serial.println(distance);
  
  // Update LCD display at certain intervals to prevent flicker
  if (millis() - lastDisplayUpdate > DISPLAY_UPDATE_INTERVAL) {
    updateLCDDisplay();
    lastDisplayUpdate = millis();
  }
  
  // Check if distance is less than threshold
  if (distance < DISTANCE_THRESHOLD && distance > 0) {
    // Activate buzzer for warning
    int buzzInterval = map(distance, 0, DISTANCE_THRESHOLD, 50, 500);
    digitalWrite(buzzerPin, HIGH);
    delay(50);
    digitalWrite(buzzerPin, LOW);
    delay(buzzInterval);
    
    // Set alert message based on distance
    if (distance < 10) {
      alertMessage = "CRITICAL!";
    } else if (distance < 25) {
      alertMessage = "WARNING!";
    } else {
      alertMessage = "CAUTION";
    }
    
    // Send alert message to Python
    Serial.println("ALERT:COLLISION_RISK");
  } else {
    // No risk detected from ultrasonic sensor
    digitalWrite(buzzerPin, LOW);
    alertMessage = "Safe Distance";
  }
  
  // Check for commands from Python
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "BUZZ") {
      // Python detected object - activate buzzer and update LCD
      for (int i = 0; i < 3; i++) {
        digitalWrite(buzzerPin, HIGH);
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print("PYTHON ALERT!");
        lcd.setCursor(0, 1);
        lcd.print("Object Detected");
        delay(200);
        digitalWrite(buzzerPin, LOW);
        delay(100);
      }
      delay(1000); // Keep message displayed for a moment
    }
  }
  
  // Wait before next measurement
  delay(MEASUREMENT_INTERVAL);
}

// Function to update the LCD display
void updateLCDDisplay() {
  lcd.clear();
  
  // First line: Distance
  lcd.setCursor(0, 0);
  lcd.print("Dist: ");
  if (distance < 1000 && distance > 0) {
    lcd.print(distance);
    lcd.print(" cm");
  } else {
    lcd.print("Out of range");
  }
  
  // Second line: Status message
  lcd.setCursor(0, 1);
  lcd.print(alertMessage);
  
  // Add visual indicators based on distance
  if (distance < DISTANCE_THRESHOLD && distance > 0) {
    // Number of warning indicators based on proximity
    int indicators = map(distance, 0, DISTANCE_THRESHOLD, 5, 0);
    lcd.setCursor(11, 1);
    for (int i = 0; i < indicators; i++) {
      lcd.print("!");
    }
  }
}