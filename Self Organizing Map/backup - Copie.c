#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.1415
#define Width 25

double *SOM[Width][Width];
int labels[Width][Width];
int n_features;

float alpha0 = 1.0;
float sigma0 = 2.5;
float lambda = 0.01;
float beta = 0.01;

char test[100] = "data";
char file_name[100] = "data";
int nb_iterations = 100;

void init(FILE *file){
    int i,j,k;
    int nb_lines;
    fscanf(file, "%d", &nb_lines);
    fscanf(file, "%d", &n_features);
    for(i=0;i<Width;i++){
        for(j=0;j<Width;j++){
            fscanf(file, "%d", &labels[i][j]);
            SOM[i][j]=(double *)(malloc(n_features * sizeof(double)));
            for(k=0;k<n_features;k++){
                fscanf(file, "%lf", &SOM[i][j][k]);
                //printf("%lf ",SOM[i][j][k]);
            }
        }
    }
}

void find_BMU(double *X, int label, int *l, int *c, int test){
    int i, j, k;
    double min_dist = 0;
    for(i = 0; i < n_features; i++)
        min_dist += (X[i] - SOM[5][5][i]) * (X[i] - SOM[5][5][i]);
    *l = 0; *c = 0;
    for(i = 0; i < Width; i++){
        for(j = 0; j < Width; j++){
            double cur_dist = 0;
            for(k = 0; k < n_features; k++)
                cur_dist += (X[k] - SOM[i][j][k]) * (X[k] - SOM[i][j][k]);
            if(cur_dist < min_dist){
                *l = i;
                *c = j;
                min_dist = cur_dist;
            }
        }
    }
    //printf("the cordinates are %d %d \n",*l,*c);
    if(!test)
        labels[*l][*c] = label;
        //printf("The label is : %d \n",labels[*l][*c]);
}

/*/
double sigma(int t){
    double res = 1 / log(log(t+15)) - 2;
    return res;
}
/*/

double sigma(int epoch){
    double res = sigma0*exp(-epoch*beta);
    return res;
}

/*/
double alpha(int t){
    double res = 1 / log(log(t+15)) - 2;
    return res;
}
/*/

double alpha(int epoch){
    double res = alpha0*exp(-epoch*lambda);
    return res;
}

/*/
double gamma_fct(int l, int c, int i, int j, int t){


    double sig = sigma(t);
    int d_2= (j - c) * (j - c) + (i - l) * (i - l);
    double res = (1 / (sig * sqrt(2 * PI))) * exp(-d_2 / (2 * sig * sig));
    return res;
}
/*/

double gamma_fct(int l, int c, int i, int j, int epoch){
    double sig = sigma(epoch);
    int d_2= (j - c) * (j - c) + (i - l) * (i - l);
    double res = (1 / (sig * sqrt(2 * PI))) * exp(-d_2 / (2 * sig * sig));
    return res;
}

void update_neuron(double *X, int l, int c, int epoch){
    int i,j,k;
    for(i=0;i<Width;i++){
        for(j=0;j<Width;j++){
            for(k=0;k<n_features;k++)
                //printf("alpha : %f, sigma : %f \n",alpha(epoch),gamma_fct(l,c,i,j,epoch));
                SOM[i][j][k] += alpha(epoch)*gamma_fct(l,c,i,j,epoch)*(X[k]-SOM[i][j][k]);
        }
    }
}


void update_SOM(FILE *file,int epoch){
    int nb_lines;
    int i, t;
    int label;
    fscanf(file, "%d", &nb_lines);
    fscanf(file, "%d", &n_features);
    double* X;
    X = (double*)malloc(n_features * sizeof(double));
    //init(file);
    for(t = 1; t <= nb_lines; t++){
        fscanf(file, "%d", &label);
        for(i = 0; i < n_features; i++)
            fscanf(file, "%lf", &X[i]);
        int l = 0, c = 0;
        find_BMU(X,label,&l,&c,0);
        update_neuron(X,l,c,epoch);
        //printf("example %d at epoch %d \n",t,epoch);
        //printf("alpha : %f, sigma : %f \n",alpha(epoch),gamma_fct(l,c,5,5,epoch));
    }
}


double accuracy(FILE* file){
    int nb_test;
    int i, t;
    int label;
    int nb_correct = 0;
    double res = 0.0;
    fscanf(file, "%d", &nb_test);
    fscanf(file, "%d", &n_features);
    double* X;
    X = (double*)malloc(n_features * sizeof(double));
    for(t = 1; t <= nb_test; t++){
        fscanf(file, "%d", &label);
        for(i = 0; i < n_features; i++)
            fscanf(file, "%lf", &X[i]);
        int l = 0, c = 0;
        find_BMU(X,label,&l,&c,1);
        //printf("True label : %d \n",label);
        //printf("Predicted label : %d \n",labels[l][c]);
        if(labels[l][c] == label)
            nb_correct++;
    }
    res = nb_correct / (double)nb_test;
    return res * 100.0;
}


void show_SOM() {
    FILE* file = fopen("carte.txt","w");
    int i,k,j;
    for(i=0; i<Width; i++) {
        for(j=0; j<Width; j++) {
            for(k=0; k<n_features; k++)
                fprintf(file, "%lf ", SOM[i][j][k]);
            fprintf(file,"\n");
        }
    }
    fclose(file);
}

void show_labels() {
    FILE* file = fopen("carte_cat.txt","w");
    int i,j;
    fprintf(file, "%d\n", Width);
    for(i=0; i<Width; i++){
        for(j=0; j<Width; j++)
            fprintf(file, "%d ", labels[i][j]);
        fprintf(file, "\n");
    }
    fclose(file);
}


int epoch;

int main()
{
    double precision;
    //double precisions[nb_iterations];
    //double min_precision = 0;
    //double min epoch;

    strcat(file_name, ".txt");
    FILE* file = fopen(file_name,"r");
    init(file);
    for(epoch=0;epoch<nb_iterations;epoch++){
        printf("apprentissage en cours, iteration: %d/%d\n", epoch+1,nb_iterations);
        FILE* file = fopen(file_name,"r");
        update_SOM(file,epoch);
        FILE* filetest = fopen(file_name,"r");
        precision = accuracy(filetest);
        printf("Precision: %lf\n", precision);
        //precisions[epoch] = precision;

    }

    show_labels();
    //show_SOM();

    strcat(test, ".txt");
    FILE* test_file = fopen(test,"r");
    precision = accuracy(test_file);
    printf("\nPrecision: %lf\n", precision);
    return 0;
}




