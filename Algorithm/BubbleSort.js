function bubleSort(arr){
    const n = arr.lenght
    for (let i = 0; i < n; i++){
    for (let j = 0; j < n - i - 1; j++ ){
        if (arr[j] > arr[j + 1]) {


            const temp = arr[j];
            arr[j] = arr[j + 1];
            arr[j + 1] = temp;


            }
        }
    }
    return arr;
}

const myArray = [5, 3, 8, 1, 2]
console.log(bublesors(myArray));