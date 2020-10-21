/*****
 * Project 02: Wheel of Fortune
 * COSC 208, Introduction to Computer Systems, Fall 2020
 *****/

//Wheel functions, linked list struct, and constants were supplied

/*
 * Adds an item to the front of the linked list of words.
 * @param list_head the first item in the list; NULL if the list is empty
 * @param word the word to add
 * @return the added item (i.e., the new first item in the list)
 */
item_t *prepend_item(item_t *list_head, char *word) {
    item_t *new_head = malloc(sizeof(item_t));
    if (new_head == NULL) {
        return NULL;
    }
    new_head->word = word;
    new_head->next = list_head;
    return new_head;
}

/*
 * Loads a list of words from a file into a linked list. Words containing 
 * non-alpha characters are ignored. All words are stored in upper-case.
 * @param filepath path to the file of words
 * @param words_loaded populated with the number of words loaded from the file
 * @return the linked list of words; NULL if an error occurred
 */
item_t *load_words(const char *filepath, int *words_loaded) {
    item_t *return_list = malloc(sizeof(item_t));
    if (return_list == NULL) {
        printf("Return is NULL");
        return NULL;
    }
    FILE *file = fopen(filepath, "r");
    if (file == NULL) {
        printf("File is NULL");
        fclose(file);
        return NULL;
    }               
    while (1) {
        char *curr_word = malloc(sizeof(char) * MAX_WORD_LENGTH);
        if (curr_word == NULL) {
            printf("Word is NULL");
            return NULL;
        }
        curr_word[0] = '\0';
        fgets(curr_word, MAX_WORD_LENGTH, file);
        if (curr_word[0] == '\0') { //Represents end of file
            free(curr_word);
            break;
        }
        if (curr_word == NULL) {
            printf("Word is NULL");
            fclose(file);
            return NULL;
        } 

        int i = 0;
        while (curr_word[i]) { //Turns word uppercase and strips newline
            if (curr_word[i] == '\n') {
                curr_word[i] = '\0';
                break;
            }
            else if (isalpha(curr_word[i])){
                curr_word[i] = toupper(curr_word[i]);
                i++;
            }
            else {
                free(curr_word);
                curr_word = NULL;
                break;
            }
        }
        if (curr_word != NULL) { //If word meets criteria
            return_list = prepend_item(return_list, curr_word);
            if (return_list == NULL) { //In case of malloc failure in call
                printf("New list head is NULL");
                fclose(file);
                return NULL;
            }
            *words_loaded = *words_loaded + 1;
        }
        else {
            free(curr_word);
        }
    }
    fclose(file);
    return return_list;
}

/*
 * Destroys a linked list and frees all memory it was using.
 * @param list the first item in the list; NULL if the list is empty
 */
void free_words(item_t *list_head) {
    item_t *last;
    while (list_head != NULL) {
        free(list_head->word);
        last = list_head;
        list_head = list_head->next;
        free(last);
    }
}

/*
 * Chooses a random word from a linked list of words.
 * @param list_head the first item in the list; NULL if the list is empty
 * @param length the number of words in the list
 * @return the chosen word; NULL if an error occurred
 */
char *choose_random_word(item_t *list_head, int length) {
    if (list_head == NULL) {
        return NULL;
    }
    int word_num = random() % length;
    item_t curr = *list_head; //Not a pointer so it doesn't edit the list
    for (int i = 0; i < word_num; i++) {
        curr = *(curr.next);
    }
    return curr.word;
}

/*
 * Initialized every value of array to ' '
 * @param arr the word to be initialized
 * @param length the number of chars in the arr
 */
void initializeWord(char *arr, int length) {
    for (int i = 0; i < length; i++) {
        arr[i] = ' ';
    }
}

/*
* Checks to see if a value is in an array. 
* @param arr array to check contents of
* @param c character to check for
* @return true if c is in arr, false if not
*/
bool isIn(char *arr, char c) {
    for (int i = 0; i < strlen(arr); i++) {
        if (arr[i] == toupper(c)) {
            return true;
        }
    }
    return false;
}

/*
* Gets user input and returns the character for the action ('1'-'5', 'B', or 'S')
* @return action character
*/
char getInput() {
    char input[MAX_INPUT_LENGTH];
    fgets(input, MAX_INPUT_LENGTH, stdin);
    if (strlen(input) != 2) {
        return '\0';
    }
    return input[0];
}

/*
* removes letter from Vowels or Consonants, or prints ABSENT_LETTER prompt if absent
* @param array vowel or consonant array pointer
* @param letter char to be searched for
*/
void updateVowOrCon(char *array, char letter) {
    for(int i = 0; i < strlen(array); i++) {
        if (array[i] == toupper(letter)) {
            array[i] = ' ';
        }
    }
}

/*
* Prints current state of solution and returns number of times letter is seen in word
* @param letter vowel or consonant guessed by player
* @param solved array pointer to current solution
* @param word array pointer to correct word
* @return times letter seen in word
*/
int handleCurrentSolved(char letter, char *solved, const char *word) {
    int seen = 0;
    for (int i = 0; i < strlen(word); i++) {
        if (word[i] == toupper(letter)) {
            seen++;
            solved[i] = toupper(letter);
        }
        if (letter == '\0') {
            printf("%c ", solved[i]);
        }
    }
    printf("\n");
    return seen;
}

/*
* Prints the current player, their earnings, and current state of solution
* @param earnings array of player earnings
* @param player index of current player
* @param solved array pointer to current solution
* @param word array pointer to correct word
* @return times letter seen in word
*/
void printGameStatus(int *earnings, int player, char* solved, const char *word) {
    printf("\nPlayer %d's turn\n", player+1);
    printf("\nPlayer %d's earnings: %d\n", player+1, earnings[player]);
    handleCurrentSolved('\0', solved, word); //Prints without updating solved
}

/*
* Prints remaining contents of vowels or consonants
* @param array pointer to either vowel or consonant array
*/
void printVowOrCon(char *array) {
    for(int i = 0; i < strlen(array); i++) {
        if(array[i] != ' ') {
            printf("%c ", array[i]);
        }
    }
    printf("\n");
}

/*
* Spins wheel and uses player input to update earnings and solution state
* @param earnings pointer to current earnings of the players
* @param player index of current player 
* @param action char of selected spin strength
* @param consonants pointer to playable consonants array
* @param solved array pointer to current solution
* @param word array pointer to correct word
* @return true if letter valid and present, false if not
*/
bool spin(int *earnings, int player, char action, char *consonants, char* solved, const char* word) {
    int value = spin_wheel(action - '0');
    printf("%s\n", "Consonant?");
    printVowOrCon(consonants);
    char choice = getInput();
    if (choice != ' ' && isIn(consonants, choice)) {
        updateVowOrCon(consonants, choice);
        int seen = handleCurrentSolved(choice, solved, word);
        if (seen == 0) {
            printf("%s\n", ABSENT_LETTER);
            printGameStatus(earnings, player, solved, word);
            return false;
        }
        earnings[player] += value * seen;
        printGameStatus(earnings, player, solved, word);
        return true;
    }
    else {
        printf("%s\n", INVALID_LETTER);
        printGameStatus(earnings, player, solved, word);
        return false;
    }
}

/*
* Allows player to select vowel if they have funds, and updates if valid choice
* @param earnings pointer to current earnings of the players
* @param player index of current player 
* @param vowels pointer to playable vowels array
* @param solved array pointer to current solution
* @param word array pointer to correct word
* @return true if insufficient funds or vowel present, false if not
*/
bool buy(int *earnings, int player, char *vowels, char* solved, const char* word) {
    if (earnings[player] > VOWEL_COST) { 
        printf("%s\n", "Vowel?");
        printVowOrCon(vowels);
        char choice = getInput();
        if (choice != ' ' && isIn(vowels, choice))  {
            earnings[player] -= VOWEL_COST;
            updateVowOrCon(vowels, choice);
            if (!handleCurrentSolved(choice, solved, word)){
                printf("%s\n", ABSENT_LETTER);
                printGameStatus(earnings, player, solved, word);
                return false;
            }
            printGameStatus(earnings, player, solved, word);
            return true;
        }
        else {
            printf("%s\n", INVALID_LETTER);
            printGameStatus(earnings, player, solved, word);
            return false;
        }
    }
    else {
        printf("%s\n", INSUFFICIENT_FUNDS);
        printGameStatus(earnings, player, solved, word);
        return true;
    }
}

/*
* Prompts user for solution and returns whether correct or not
* @param finalSolve array pointer to player guess (reusable dedicated array)
* @param word array pointer to correct word
* @return true if guess is correct, false if not
*/
bool solve(char *finalSolve, const char* word) {
    printf("%s", SOLUTION_PROMPT);
    fgets(finalSolve, MAX_INPUT_LENGTH, stdin); 
    finalSolve[strlen(finalSolve)-1] = '\0'; //Strips newline from end
    for (int j = 0; j < strlen(finalSolve); j++) {
        finalSolve[j] = toupper(finalSolve[j]);
    }
    if (strcmp(finalSolve, word) == 0) {
        return true;
    }
    else {
        printf("%s\n", INCORRECT_SOLUTION);
        return false;
    }
}

/*
 * Play a single round of wheel of fortune.
 * @param word array pointer of the word to be guessed
 * @return the number of the player who won
 */
int play_round(const char *word) {

    int player = 0;

    char solved[strlen(word)];
    initializeWord(solved, strlen(word));
    char finalSolve[MAX_WORD_LENGTH];
    initializeWord(finalSolve, MAX_WORD_LENGTH);

    int earnings[NUM_PLAYERS];
    for (int i = 0; i < NUM_PLAYERS; i++) {
        earnings[i] = 0;
    }

    char *consonants = makeConsonants(); 
    char *vowels = makeVowels(); 
    
    //Create initial underscore array
    for(int i = 0; i < strlen(word); i++){
        solved[i] = '_';
    }

    int round_ct = 0;
    while (1) {

        player = round_ct%2;

        printGameStatus(earnings, player, solved, word);

        bool ongoing = true;
        while (ongoing) {
            printf(ACTION_PROMPT);
            char action = getInput();
            if (action >= '1' && action <= '5') { 
                ongoing = spin(earnings, player, action, consonants, solved, word);
            }
            else if (toupper(action) == 'B') {
                ongoing = buy(earnings, player, vowels, solved, word);
            }
            else if (toupper(action) == 'S') {
                if (solve(finalSolve, word)) {
                    free(consonants);
                    free(vowels);
                    return player+1;
                }
                ongoing = false;
            }
            else {
                printf("%s\n", INVALID_ACTION);
            }
        }
        round_ct++;

    }

}


/*
 * Play wheel of fortune using words loaded from a file.
 */
int main() {
    //Initialize wheel
    initialize_wheel();

    //Select random seed
    srandom(time(0)); //Comment for deterministic words selection

    //Load words
    int numwords = 0;
    item_t *list_head = load_words(WORDS_FILE, &numwords);
    if (list_head == NULL) {
        printf("Failed to load words from %s\n", WORDS_FILE);
        return 1;
    }

    //Select a word
    char *word = choose_random_word(list_head, numwords);
    printf("WORD IS %s", word);
    if (word == NULL) {
        printf("Failed to choose a word\n");
        return 1;
    }

    //Play game
    int winner = play_round(word);
    printf("Player %d solved the puzzle!\n", winner);

    //Clean up
    free_words(list_head);

    //Clean-up wheel_rows
    for (int r = 0; r < DISPLAY_ROWS; r++) {
        free(wheel_row[r]);
    }
}
