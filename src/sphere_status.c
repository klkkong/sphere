#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>

int print_usage(FILE* stream, char* argv0, int return_status);
int open_status_file(char* cwd, char* sim_name, int format);

int main(int argc, char *argv[])
{

    // Read path to current working directory
    char *cwd;
    cwd = getcwd(0, 0);
    if (!cwd) {  // Terminate program execution if path is not obtained
        perror("Could not read path to current workind directory "
                "(getcwd failed)");
        return 1;
    }

    if (argc == 1) {
        DIR *dir;
        struct dirent *ent;
        char outputdir[1000];
        char* dotpos;
        sprintf(outputdir, "%s/output/", cwd);
        if ((dir = opendir(outputdir)) != NULL) {
            puts("Simulations with the following ID's are found in the "
                    "./output/ folder:");

            while ((ent = readdir(dir)) != NULL) {
                if ((dotpos = strstr(ent->d_name, ".status.dat")) != NULL) {
                    *dotpos = '\0';
                    printf("\t%s\t(", ent->d_name);
                    (void)open_status_file(cwd, ent->d_name, 1);
                    puts(")");

                }
            }
            closedir(dir);
        } else {
            fprintf(stderr, "Error: could not open directory: %s\n", outputdir);
            return 1;
        }
        return 0;

    } else if (argc != 2) {
        return print_usage(stderr, argv[0], 1);
    } else if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        return print_usage(stdout, argv[0], 0);
    }

    return open_status_file(cwd, argv[1], 0);
}

int print_usage(FILE* stream, char* argv0, int return_status)
{
    fprintf(stream, "sphere simulation status checker. Usage:\n"
            "%s [simulation id]\n"
            "If the simulation id isn't specified, a list of simulations \n"
            "found in the ./output/ folder will be shown\n", argv0);
    return return_status;
}


int open_status_file(char* cwd, char* sim_name, int format) {
    // Open the simulation status file
    FILE *fp;
    char file[1000]; // Complete file path+name variable
    sprintf(file,"%s/output/%s.status.dat", cwd, sim_name);

    if ((fp = fopen(file, "rt"))) {
        float time_current;
        float time_percentage;
        unsigned int file_nr;

        if (fscanf(fp, "%f%f%d", &time_current, &time_percentage, &file_nr)
                != 3) {
            fprintf(stderr, "Error: could not parse file %s\n", file);
            return 1;
        }

        if (format == 1) {
            printf("%.2f s / %.0f %% / %d",
                    time_current, time_percentage, file_nr);
        } else {
            printf("Reading %s:\n"
                    " - Current simulation time:  %f s\n"
                    " - Percentage completed:     %f %%\n"
                    " - Latest output file:       %s.output%05d.bin\n",
                    file, time_current, time_percentage, sim_name, file_nr);
        }

        fclose(fp);

        return 0; // Exit program successfully

    } else {
        fprintf(stderr, "Error: Could not open file %s\n", file);
        return 1;
    }
}

// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
