#ifndef __ASYNC__
#define __ASYNC__

#include <pthread.h>

typedef struct my_item {
  struct my_item *next;
  struct my_item *prev;
  void (*function)(int arg);
  int arg;
} my_item_t;

typedef struct my_semaphore {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int v;
} my_semaphore;

typedef struct my_queue {
  int size;
  my_item_t *head;
} my_queue_t;

void async_init(int);
void async_run(void (*fx)(int), int args);

#endif
