pinMode(GREEN_LED, OUTPUT);

    Serial.println("Calibrating MQ135 sensor...");
    baselineValue = calibrateMQ135();
    Serial.print("Baseline Air Quality: ");
    Serial.println(baselineValue);
}

void loop() {
    int airQuality = getAverageReading();

    Serial.print("Air Quality: ");
    Serial.println(airQuality);

    // Adjust threshold dynamically
    int pollutionThreshold = baselineValue + 100;

    if (airQuality > pollutionThreshold) {  
        digitalWrite(RED_LED, HIGH);
        digitalWrite(GREEN_LED, LOW);
        Serial.println("High Pollution Detected!");
        espSerial.println("HIGH_POLLUTION");
    } else {  
        digitalWrite(RED_LED, LOW);
        digitalWrite(GREEN_LED, HIGH);
        Serial.println("Air Quality Good.");
        espSerial.println("GOOD_AIR");
    }

    delay(2000); 
}

// Function to calibrate MQ135 by taking an average baseline reading
int calibrateMQ135() {
    int sum = 0;
    for (int i = 0; i < sampleSize; i++) {
        sum += analogRead(MQ135_PIN);
        delay(100);
    }
    return sum / sampleSize;
}

// Function to get the average air quality reading
int getAverageReading() {
    int sum = 0;
    for (int i = 0; i < sampleSize; i++) {
        sum += analogRead(MQ135_PIN);
        delay(100);
    }
    return sum / sampleSize;
}