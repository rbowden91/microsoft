#include <stdio.h>
#include <cs50.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

int     main(int argc, string argv[])
{
        if (argc != 2)
        {
            printf ("usage : ./vigenere + key \n");
            return 1;
        }
        
        string key = argv[1];
        int f = 0;
        int g = strlen(key);
        
        while (f < g)
        {
            if (!isalpha(key[f]))
            {
                printf ("error!\n");
                return 1;
            }   
            f++;
        }
        
        string text = GetString();
        int i = 0;
        int j = 0;
        int fuck;
        int k;
        int result;
        int n = strlen(text);
        int ne = strlen (key);
    
        while (i < n)
        {
                j = 0;
               while (j < ne && i < n)
               {
                   if (key[j] >= 'a' && key[j] <= 'z')
                   {
                       if ((text[i] >= 'a' && text[i] <= 'z') || (text[i] >= 'A' && text[i] <= 'Z'))
                       {
                           k = key[j] - 97;
                           result = text[i] + k;
                           if (result > 122)
                            {
                                fuck = 123 - text[i];
                                k  = k - fuck;
                                result = 97 + k;
                                printf("%c", result);
                            }  
                            else
                                printf("%c", result);
                           j++;
                           i++;
                       }
                       else
                       {
                           printf ("%c", text[i]);
                            i++;
                       }
                        
                   }
                   else if (key[j] >= 'A' && key[j] <= 'Z')
                   {
                       if ((text[i] >= 'a' && text[i] <= 'z') || (text[i] >= 'A' && text[i] <= 'Z'))
                       {
                           k = key[j] - 65;
                           result = text[i] + k;
                           if (result > 90)
                            {
                                fuck = 91 - text[i];
                                k  = k - fuck;
                                result = 65 + k;
                                printf("%c", result);
                             }
                             else
                                printf("%c", result);
                           j++;
                           i++;
                       }
                      /* else
                       {
                           printf ("%c", text[i]);
                            i++;
                       }*/
                   }
                   
               }
        }
        printf("\n");
}