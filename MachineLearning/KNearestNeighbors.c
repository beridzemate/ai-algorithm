#include <stdlib.h>

typedef struct {
    double x;
    double y;
    int label;
} Point;

int compare(const void *a, const void *b) {
    double distA = ((Point *)a)->x;
    double distB = ((Point *)b)->x;
    return (distA > distB) - (distA < distB);
}


double euclidean_distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}


int knn(Point data[], int dataSize, Point query, int k) {

    for (int i = 0; i < dataSize; i++) {
        data[i].x = euclidean_distance(query.x, query.y, data[i].x, data[i].y);
    }


    qsort(data, dataSize, sizeof(Point), compare);

 
    int labels[2] = {0};
    for (int i = 0; i < k; i++) {
        labels[data[i].label]++;
    }

   
    return labels[0] > labels[1] ? 0 : 1;
}
