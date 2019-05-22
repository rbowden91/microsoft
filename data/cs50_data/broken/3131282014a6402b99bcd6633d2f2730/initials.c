#include <stdio.h>
#include <cs50.h>
#include <string.h>
#include <ctype.h>

int main(void)
{
    string name = GetString();//get name
    
    if (name != NULL)//check error
    {
        printf("%c", toupper(name[0]));
        for (int i = 0, n = strlen(name); i <= n; i++)
        {
            if (name[i] == ' ')//space
                printf("%c", toupper(name[i + 1]));//printf and toupper
        }

        printf("\n");//newline
    }
}