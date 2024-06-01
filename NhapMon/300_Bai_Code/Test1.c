#include <stdio.h>
#include <math.h>
int main (){
    double S, R, V;

    printf("Nhap vao diem tich hinh cau: ");
    scanf("%lg", &S);
     
    // Tin ban kinh thong qua dien tich S
    R = sqrt(S/(4*3.14));

    // Tinh the tich hinh cau 
    V = (4/3)*3.14*R*R*R;

    // In ra gia the tich hinh tron
    printf("The tich hinh cau la: %g", V);
}
