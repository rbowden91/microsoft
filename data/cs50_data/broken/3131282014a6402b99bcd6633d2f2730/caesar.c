#include <stdio.h>
#include <string.h>
#include <cs50.h>
#include <ctype.h>

int main(int argc,string argv[])
{
    if(argc != 2 )//check error
    {
    printf("Usage: ./caesar key");//error msg
    return 1;
    }
    int k = atoi(argv[1]);// change to int
    char* input = GetString();// get input from user
for (int i = 0, n = strlen(input); i < n; i++)
    {
        if (isalpha(input[i]))//check alphabetic letter
        {
            if (isupper(input[i]))//uppercase
            {
                int result = (input[i] + k - 65) % 26 + 65;
                printf("%c", result);
            }
            else
            {
                int result = (input[i] + k - 97) % 26 + 97;//lowercase
                printf("%c", result);
            }
        }
        else
            printf("%c", input[i]);
    }
    // print new line
    printf("\n");
   
}

