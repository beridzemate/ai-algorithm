function calculateWage(totalHours, hourlyRate, threshold = 150, overtimeCoef = 1.5) {
    let totalWage;

    if (totalHours > threshold) {
        const regularHours = threshold;
        const overtimeHours = totalHours - threshold;
        
        totalWage = (regularHours * hourlyRate) + 
                    (overtimeHours * hourlyRate * overtimeCoef);
    } else {
        totalWage = totalHours * hourlyRate;
    }

    return totalWage;
}

// Example Usage
const totalHours = 160;  // Enter total hours worked
const hourlyRate = 10;    // Enter hourly wage rate

const wage = calculateWage(totalHours, hourlyRate);
console.log(`Total wage: $${wage}`);
