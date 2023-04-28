#include <curses.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#include <cstdlib>

#define ROW 10
#define COLUMN 50
#define _COLUMN (COLUMN - 1)
#define THREAD_NUM (ROW - 1)

bool stop = false;
bool q = false;
pthread_mutex_t mutex;
int log_ids[THREAD_NUM];
int flag[THREAD_NUM + 1];
int count = 0;

struct Node {
  int x, y;
  Node(int _x, int _y) : x(_x), y(_y){};
  Node(){};
} frog;

char map[ROW + 10][COLUMN];
enum Status { WIN, LOSE, NORMAL };

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0.
int kbhit(void) {
  struct termios oldt, newt;
  int ch;
  int oldf;

  tcgetattr(STDIN_FILENO, &oldt);

  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);

  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

  ch = getchar();

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);

  if (ch != EOF) {
    ungetc(ch, stdin);
    return 1;
  }
  return 0;
}

int checkFrogPosition(int x, int y) {
  if (y < 0 || y >= _COLUMN) return LOSE;
  if (x == 0) return WIN;
  if (x == ROW) return NORMAL;
  // judges in the logs or not
  if (map[x][y] == '=') return NORMAL;
  return LOSE;
}

void *logs_move(void *t) {
  // dereference the log_id
  int log_id = *((int *)t);

  // random birth position
  srand(time(NULL) + log_id);

  // get left side and right side
  int left = rand() % _COLUMN;
  int right = (left + rand() % 6 + 10) % _COLUMN;
  while (!stop) {
    usleep(1000 * 100);
    pthread_mutex_lock(&mutex);
    if (flag[log_id] != 0) {
      pthread_mutex_unlock(&mutex);
      continue;
    }
    flag[log_id] = 1;
    /*  Move the logs  */
    int delta;
    if (log_id & 1) {
      // move left
      delta = -1;
    } else {
      // move right
      delta = 1;
    }

    // clear row ' '
    for (int i = 0; i < _COLUMN; i++) {
      map[log_id][i] = ' ';
    }
    // move logs
    left = (left + delta + _COLUMN) % _COLUMN;
    right = (right + delta + _COLUMN) % _COLUMN;
    for (int i = left; i != right; i = (i + 1) % _COLUMN) {
      map[log_id][i] = '=';
    }
    map[log_id][right] = '=';

    // if frog in this log, move it with log
    if (frog.x == log_id) {
      frog.y += delta;
    }
    if (frog.y < 0 || frog.y > _COLUMN) {
      stop = true;
    }

    // maybe frog move from river bank, so should reassign it
    int j;
    for (j = 0; j < COLUMN - 1; ++j) map[ROW][j] = map[0][j] = '|';

    for (j = 0; j < COLUMN - 1; ++j) map[0][j] = map[0][j] = '|';

    /*  Check keyboard hits, to change frog's position or quit the game. */
    if (kbhit()) {
      char kb = getchar();
      switch (kb) {
        case 'w':
        case 'W':
          frog.x--;
          if (checkFrogPosition(frog.x, frog.y) != NORMAL) {
            stop = true;
          }
          break;
        case 'a':
        case 'A':
          frog.y--;
          if (checkFrogPosition(frog.x, frog.y) != NORMAL) {
            stop = true;
          }
          break;
        case 's':
        case 'S':
          frog.x++;
          if (checkFrogPosition(frog.x, frog.y) != NORMAL) {
            stop = true;
          }
          break;
        case 'd':
        case 'D':
          frog.y++;
          if (checkFrogPosition(frog.x, frog.y) != NORMAL) {
            stop = true;
          }
          break;
        case 'q':
        case 'Q':
          stop = true;
          q = true;
          break;
        default:;
      }
    }
    map[frog.x][frog.y] = '0';

    /*  Check game's status  */
    if (stop) {
      pthread_mutex_unlock(&mutex);
      break;
    }
    count++;

    if (count == THREAD_NUM) {
      memset(flag, 0, sizeof(flag));
      /*  Print the map on the screen  */
      puts("\033[H\033[2J");  // first clear window
      for (int i = 0; i <= ROW; i++) {
        puts(map[i]);
      }
      count = 0;
    }

    pthread_mutex_unlock(&mutex);
  }

  return nullptr;
}

int main(int argc, char *argv[]) {
  // Initialize the river map and frog's starting position
  memset(map, 0, sizeof(map));
  int i, j;
  for (i = 1; i < ROW; ++i) {
    for (j = 0; j < COLUMN - 1; ++j) map[i][j] = ' ';
  }

  for (j = 0; j < COLUMN - 1; ++j) map[ROW][j] = map[0][j] = '|';

  for (j = 0; j < COLUMN - 1; ++j) map[0][j] = map[0][j] = '|';

  frog = Node(ROW, (COLUMN - 1) / 2);
  map[frog.x][frog.y] = '0';

  // Print the map into screen
  for (i = 0; i <= ROW; ++i) puts(map[i]);

  /*  Create pthreads for wood move and frog control.  */

  memset(flag, 0, sizeof(flag));

  // initialize the mutex referenced by mutex with default attributes
  // also check if the mutex is valid (return value is 0 if valid)
  if (pthread_mutex_init(&mutex, NULL) != 0) {
    printf("Failed to initialize mutex lock!");
  }

  // declare identifiers for threads
  pthread_t threads[THREAD_NUM];

  // create threads for logs
  for (int i = 0; i < THREAD_NUM; i++) {
    log_ids[i] = i + 1;
    pthread_create(&threads[i], NULL, logs_move, &log_ids[i]);
  }

  for (int i = 0; i < THREAD_NUM; i++) {
    pthread_join(threads[i], NULL);
  }

  /*  Display the output for user: win, lose or quit.  */
  puts("\033[H\033[2J");
  if (q) {
    printf("You exit the game.\n");
  } else if (checkFrogPosition(frog.x, frog.y) == WIN) {
    printf("You win the game!!\n");
  } else {
    printf("You lose the game!!\n");
  }
  pthread_mutex_destroy(&mutex);

  return 0;
}
