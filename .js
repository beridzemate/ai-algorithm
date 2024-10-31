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


const totalHours = 160;  
const hourlyRate = 10;    

const wage = calculateWage(totalHours, hourlyRate);
console.log(`Total wage: $${wage}`);
