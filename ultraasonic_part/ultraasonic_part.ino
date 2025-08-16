// Define pins for the ultrasonic sensors and buzzer
const int trigPinFront = 6; // Front sensor trigger pin
const int echoPinFront = 7; // Front sensor echo pin
const int trigPinLeft = 9;  // Left sensor trigger pin
const int echoPinLeft = 10;  // Left sensor echo pin
const int trigPinRight = 11; // Right sensor trigger pin
const int echoPinRight = 12; // Right sensor echo pin
const int buzzerPin = 8;     // Buzzer pin

// Threshold distance for obstacle detection (in cm)
const int thresholdDistance = 20; // Set to 15 cm

void setup() {
  Serial.begin(9600); // Initialize serial communication
  pinMode(trigPinFront, OUTPUT); // Set front trigger pin as output
  pinMode(echoPinFront, INPUT);   // Set front echo pin as input
  pinMode(trigPinLeft, OUTPUT);    // Set left trigger pin as output
  pinMode(echoPinLeft, INPUT);     // Set left echo pin as input
  pinMode(trigPinRight, OUTPUT);   // Set right trigger pin as output
  pinMode(echoPinRight, INPUT);    // Set right echo pin as input
  pinMode(buzzerPin, OUTPUT);       // Set buzzer pin as output
}

void loop() {
  // Measure distance using the front, left, and right sensors
  int distanceFront = getDistance(trigPinFront, echoPinFront);
  int distanceLeft = getDistance(trigPinLeft, echoPinLeft);
  int distanceRight = getDistance(trigPinRight, echoPinRight);
  
  // Print the results
  Serial.print("Front Distance: ");
  Serial.print(distanceFront);
  Serial.println(" cm");
  
  Serial.print("Left Distance: ");
  Serial.print(distanceLeft);
  Serial.println(" cm");
  
  Serial.print("Right Distance: ");
  Serial.print(distanceRight);
  Serial.println(" cm");

  // Activate buzzer if any distance is less than threshold
  if ((distanceFront > 0 && distanceFront < thresholdDistance) ||
      (distanceLeft > 0 && distanceLeft < thresholdDistance) ||
      (distanceRight > 0 && distanceRight < thresholdDistance)) {
      digitalWrite(buzzerPin, HIGH); // Turn on buzzer
  } else {
      digitalWrite(buzzerPin, LOW);  // Turn off buzzer
  }

  delay(1000); // Wait for 1 second before the next measurement
}

// Function to get the distance measurement from an ultrasonic sensor
int getDistance(int trigPin, int echoPin) {
  // Send a 10us pulse to trigger the sensor
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Measure the duration of the echo pulse
  long duration = pulseIn(echoPin, HIGH);
  if (duration > 0) {
      return duration * 0.034 / 2; // Return distance in cm
  } else {
      return -1; // Invalid reading
  }
}
