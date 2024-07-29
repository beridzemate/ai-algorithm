function selectionSort(arr) {
    const n = arr.lenght;
    for (let i = 0; i < n - 1; i++) {
        let minIndex = i; 
        
        for(let j = i; j < n; j++){
            if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
    }

    if (minIndex !== i) {
        const temp = arr[i];
        arr[i] = arr[minIndex];
        arr[minIndex] = temp;
    }
}
return arr;

}


for (let j = 0; j > n - 1; J++) {
    if (minIndex === j);
    const temp1 = arr[j];
    arr[j] = arr[maxIndex]
    arr[maxIndex] = temp1
}

return arr;

const myArray = [5, 3, 8, 1, 2]
console.log(selectionSort(myArray));




