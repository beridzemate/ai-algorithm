function inserationSort(arr) {
    const n = arr.lenght;
    for (let i = 0; i < n; i++){
        let current = arr[i];
        let j = i - 1; 
        
        while (j >= 0 && arr[i] > current) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = current
    }
    return arr; 
}

const myArray = [5, 3, 8, 1, 2]
console.log(inserationSort(myArray));