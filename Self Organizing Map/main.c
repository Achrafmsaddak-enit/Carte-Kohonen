#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <windows.h>
#include <dos.h>

#define PI 3.1415
#define Width 40 //largeur de la carte SOM


//-----------------------------------------------------------------------------------------------------------------------
//paramètres à manipuler :
//-----------------------------------------------------------------------------------------------------------------------
int nb_iterations = 100; //nombre d'itérations : nombre de fois la liste des données est parcourue
float alpha0 = 0.1;
float sigma0 = 7.5;
float lambda = 1.0/100; //lambda=1/nb_iterations
float beta = log(7.5)/100; //beta=log(sigma0)/nb_iterations
                           //à l'itération 'epoch=nb_iterations*(1-log(sigma)/log(sigma0))', le nombre de neurones modifiés est '(2*sigma)^2'
//-----------------------------------------------------------------------------------------------------------------------


//noms des fichiers :
char file_init[100] = "data.txt"; //nom du fichier des données utilisées pour l'initialisation
char file_data[100] = "data.txt"; //nom du fichier des données utilisées pour l'apprentissage
char file_test[100] = "test.txt"; //nom du fichier des données test

//préparation :
double *SOM[Width][Width]; //la carte SOM
int labels[Width][Width]; //la carte des catégories
int n_features; //nombre de descripteurs : donné dans le fichier des données


//-----------------------------------------------------------------------------------------------------------------------
//fonctions :
//-----------------------------------------------------------------------------------------------------------------------
void init(FILE *file){ //initialisation de la carte SOM et de la carte labels à partir du fichier file
    int i,j,k; //k parcourt les coefficients du vecteur de coordonnées (i,j) de la carte SOM
    int nb_lines; //nombre d'images données pour l'apprentissage

    //la ligne 1 du fichier est de la forme : "[nb_lines] [n_features]"
    fscanf(file, "%d", &nb_lines);
    fscanf(file, "%d", &n_features);

    for(i=0;i<Width;i++){
        for(j=0;j<Width;j++){

            //chaque ligne l >= 2 du fichier est de la forme : "[label] [feature1] [feature2] ... "

            fscanf(file, "%d", &labels[i][j]);

            SOM[i][j]=(double *)(malloc(n_features * sizeof(double))); //allocation de memoire pour la case (i,j)
            for(k=0;k<n_features;k++){
                fscanf(file, "%lf", &SOM[i][j][k]);
            }
        }
    }
}

void find_BMU(double *X, int label, int *l, int *c, int test){ //trouver le neurone gagnant et retourner ses indices (l,c) dans la carte SOM
    // X : vecteur donnée à trouver son BMU
    // test=0 --> en cours d'apprentissage
    // test=1 --> en cours de test
    // label : catégorie de X --> n'est pas utilisé si test=1

    int i, j, k; //k parcourt les coefficients du vecteur de coordonnées (i,j) de la carte SOM
    double min_dist=-1; //distance minimale entre X et les éléments de SOM

    //calcul de min_dist :
    for(i = 0; i < Width; i++){
        for(j = 0; j < Width; j++){

            //calcul de la distance cur_dist entre X et le vecteur (i,j)
            double cur_dist = 0;
            for(k = 0; k < n_features; k++)
                cur_dist += (X[k] - SOM[i][j][k]) * (X[k] - SOM[i][j][k]);

            if((cur_dist < min_dist) || (min_dist==-1)){
                *l = i;
                *c = j;
                min_dist = cur_dist;
            }
        }
    }
    if(!test)
        labels[*l][*c] = label; //mise à jour de la carte des catégories pendant l'apprentissage
}

//les fonctions sigma, alpha et gamma sont nécessaires à la mise à jour de la carte SOM :
double sigma(int epoch){
    double res = sigma0*exp(-epoch*beta);
    return res;
}
double alpha(int epoch){
    double res = alpha0*exp(-epoch*lambda);
    return res;
}
double gamma_fct(int l, int c, int i, int j, int epoch){
    //(l,c) : BMU
    //(i,j) : un voisin du BMU dans la carte SOM

    double sig = sigma(epoch); //sigma représsente la largeur de la gaussienne gamma
    int dist= (j - c) * (j - c) + (i - l) * (i - l); //distance (dans la carte) entre (i,j) et (l,c)
    double res =  exp(-dist*dist / (2 * sig*sig));
    return res;
}

void update_neuron(double *X, int l, int c, int epoch){ //mettre à jour les neurones de la carte SOM à partir d'une donnée X
    //(l,c) : BMU

    int i,j,k; //k parcourt les coefficients du vecteur de coordonnées (i,j) de la carte SOM
    int sig= (int)sigma(epoch);

    //mise a jour uniquement des neurones distants de (l,c) d'une distance <= sigma :
    for(i=(l-sig)*((l-sig)>0);i<((l+sig+1)*((l+sig+1)<Width) + (Width-1)*((l+sig+1)>Width));i++){
        for(j=(c-sig)*((c-sig)>0);j<((c+sig+1)*((c+sig+1)<Width) + (Width-1)*((c+sig+1)>Width));j++){
            for(k=0;k<n_features;k++)
                if(((j - c) * (j - c) + (i - l) * (i - l))<=(sig*sig))
                SOM[i][j][k] += alpha(epoch)*gamma_fct(l,c,i,j,epoch)*(X[k]-SOM[i][j][k]);
        }
    }
    /*/for(i=0;i<Width;i++){
        for(j=0;j<Width;j++){
            for(k=0;k<n_features;k++)
                //printf("alpha : %f, sigma : %f \n",alpha(epoch),gamma_fct(l,c,i,j,epoch));
                SOM[i][j][k] += alpha(epoch)*gamma_fct(l,c,i,j,epoch)*(X[k]-SOM[i][j][k]);
        }
    }
    /*/
}

void update_SOM(FILE *file,int epoch){ //mettre à jour les neurones de la carte SOM à partir d'un fichier de données 'file'
    int nb_lines; //nombre d'images données pour l'apprentissage

    //la ligne 1 du fichier est de la forme : "[nb_lines] [n_features]"
    fscanf(file, "%d", &nb_lines);
    fscanf(file, "%d", &n_features);

    int label; //catégorie
    double* X= (double*)malloc(n_features * sizeof(double)); //vecteur des descripteurs

    //chaque ligne l >= 2 du fichier est de la forme : "[label] [feature1] [feature2] ... "
    for(int t = 0; t < nb_lines; t++){
        fscanf(file, "%d", &label); //catégorie du vecteur X
        for(int i = 0; i < n_features; i++)
            fscanf(file, "%lf", &X[i]); //remplissage du vecteur X

        //ayant obtenu la donnée X, trouver le BMU correspondant puis mettre a jour la carte SOM :
        int l,c;
        find_BMU(X,label,&l,&c,0); // trouver le neurone gagnant (l,c)
        update_neuron(X,l,c,epoch); //mettre à jour la carte SOM
    }
}

void shuffle(){//mélanger les neurones

    srand(time(NULL ));
    for (int i=0;i<Width;i++){
        for(int j=0;j<Width;j++){
            int i_random = rand() % (Width-i) + i; //génère aléatoirement i_random dans [i,Width[
            int j_random = rand() % (Width-j) + j; //génère aléatoirement j_random dans [j,Width[

            //échanger les 2 éléments de la carte SOM
            double* tmp=SOM[i][j];
            SOM[i][j]=SOM[i_random][j_random];
            SOM[i_random][j_random]=tmp;

            //échanger les 2 éléments de la carte labels
            int tmp2=labels[i][j];
            labels[i][j]=labels[i_random][j_random];
            labels[i_random][j_random]=tmp2;
        }
    }
}


double accuracy(FILE* file){ //fait tester la precision du modèle
    int nb_test; //nombre d'images à tester et classer

    //la ligne 1 du fichier est de la forme : "[nb_lines] [n_features]"
    fscanf(file, "%d", &nb_test);
    fscanf(file, "%d", &n_features);

    //chaque ligne l >= 2 du fichier est de la forme : "[label] [feature1] [feature2] ... "
    int label;
    double* X = (double*)malloc(n_features * sizeof(double));


    //calcul du nombre d'estimations correctes de la catégorie
    int nb_correct = 0;
    for(int t = 0; t < nb_test; t++) {
        fscanf(file, "%d", &label); //label : la vraie catégorie du vecteur X
        for (int i = 0; i < n_features; i++)
            fscanf(file, "%lf", &X[i]); //remplissage du vecteur X
        int l, c;
        find_BMU(X, label, &l, &c, 1); // le neurone (l,c) trouvé donnera la catégorie de X estimée : labels[l][c]

        if (labels[l][c] == label)
            nb_correct++;
    }

    return (double) nb_correct / nb_test * 100.0; //retourner le pourcentage des estimations correctes
}

void show_SOM() { //affichage des neurones de la carte SOM dans un fichier carte.txt
    FILE* file = fopen("carte.txt","w");
    fprintf(file, "%d %d\n",Width*Width,n_features);
    for(int i=0; i<Width; i++) {
        for(int j=0; j<Width; j++) {
            for(int k=0; k<n_features; k++)
                fprintf(file, "%lf ", SOM[i][j][k]);
            fprintf(file,"\n"); //chaque neurone est représenté dans une ligne
        }
    }
    fclose(file);
}

void show_labels() { //affichage de la carte des catégories dans un fichier carte_cat.txt
    FILE* file = fopen("carte_cat.txt","w");
    fprintf(file, "%d\n", Width);
    for(int i=0; i<Width; i++){
        for(int j=0; j<Width; j++)
            fprintf(file, "%d ", labels[i][j]);
        fprintf(file, "\n");
    }
    fclose(file);
}

//-----------------------------------------------------------------------------------------------------------------------


int main()
{
    clock_t start=clock(); //initialisation du timer

    //initialisation de la carte:
    FILE* file = fopen(file_init,"r");
    init(file);
    fclose(file);

    //apprentissage :
    double precision;
    for(int epoch=0;epoch<nb_iterations;epoch++){
        printf("Apprentissage en cours, iteration : %d/%d\n", epoch+1,nb_iterations);
/*/
        //mélanger la carte
        if(epoch<=(int(nb_iterations*(1-log(2)/log(sigma0)))))
            shuffle();
/*/
        //mise à jour de la carte SOM à l'itération epoch :
        file = fopen(file_data,"r");
        update_SOM(file,epoch);
        fclose(file);

        //test de la carte obtenue à l'itération epoch (afin de voir si la précision s'améliore encore) :
            //test avec le fichier data :
        file = fopen(file_data,"r");
        precision = accuracy(file);
        fclose(file);
        printf("Precision :\n \tfichier data -----> %.3lf %%\n", precision);
            //test avec le fichier test :
        file = fopen(file_test,"r");
        precision = accuracy(file);
        fclose(file);
        printf("\tfichier test -----> %.3lf %%\n", precision);

    }

    //calcul de la durée d'apprentissage :
    clock_t duration=clock()-start;
    int hours=(duration/CLOCKS_PER_SEC)/3600; //nombre d'heures
    int minutes=((duration/CLOCKS_PER_SEC)%3600)/60; //nombre de minutes
    int secondes=(duration/CLOCKS_PER_SEC)%60; //nombre de secondes

    //stockage des résultats dans des fichiers txt :
    show_labels();
    show_SOM();

    //test du modèle final :
    FILE* test_file = fopen(file_test,"r");
    precision = accuracy(test_file);
    fclose(test_file);

    //affichage des résultats finaux :
    printf("\n----------- Fin de l'apprentissage -----------\n");
    printf("Precision : %.3lf %%\n", precision);
    printf("Duree de l'apprentissage : %dh %dmin %ds",hours,minutes,secondes);

    //faire un signal sonore à la fin du programme
    Beep(1000,3000);

    return 0;
}




