#include <cs50.h> //adds GetString(), which basically renames char to string
#include <stdio.h> //allows things like printftouch
#include <stdlib.h> //adds the atoi() function
#include <string.h> //doin stuff with strings like strlen
#include <ctype.h> //adds isupper(), islower(), and isalpha()

int main(int argc, string argv[])
{
    //defining all my variables
    string keyword;
    string inputtext;
    int j = 0;
    int kwl;

    //test for an argument
    if (argc != 2)
    {
        printf("Try again, gimme one argument of only aplha characters.\n");
        return 1;
    }

    keyword = argv[1];

    //test that argument is all alpha
    for (int i = 0, n = strlen(keyword); i < n; i++)
    {
        if (!isalpha(argv[1][i]))
        {
            printf("Try again, gimme letters only.\n");
            return 1;
        }
        else
        {
            if (isupper(keyword[i]))
            {
                keyword[i] -= 'A';
            }
            else
            {
                keyword[i] -= 'a';
            }
        }
    }

    inputtext = GetString();

    for (int i = 0, n = strlen(inputtext); i < n; i++)
    {
        if (isalpha(inputtext[i]))
        {

            if (isupper(inputtext[i]))
            {
                inputtext[i] = ((((inputtext[i] - 'A') + (keyword[j%strlen(keyword)])) % 26) +'A');
            }
            else
            {
                inputtext[i] = ((((inputtext[i] - 'a') + (keyword[j%strlen(keyword)])) % 26) + 'a');
                j++;
            }

        }
    }
    printf("%s\n", inputtext);
}
