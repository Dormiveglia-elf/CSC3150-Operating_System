
#include "async.h"
#include "utlist.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define LOG_ERROR(str) fprintf(stderr, str)

/***********************************/
/* Init semaphore to 1 or 0 */
static void semaphore_init(my_semaphore *semaphore, int value) {
  if (value < 0 || value > 1) {
    exit(1);
  }
  pthread_mutex_init(&(semaphore->mutex), NULL);
  pthread_cond_init(&(semaphore->cond), NULL);
  semaphore->v = value;
}

/* Post to at least one thread */
static void semaphore_post(my_semaphore *semaphore) {
  pthread_mutex_lock(&semaphore->mutex);
  semaphore->v = 1;
  pthread_cond_signal(&semaphore->cond);
  pthread_mutex_unlock(&semaphore->mutex);
}

/* Wait on semaphore until semaphore has value 0 */
static void semaphore_wait(my_semaphore *semaphore) {
  pthread_mutex_lock(&semaphore->mutex);
  while (semaphore->v != 1) {
    pthread_cond_wait(&semaphore->cond, &semaphore->mutex);
  }
  semaphore->v = 0;
  pthread_mutex_unlock(&semaphore->mutex);
}
/***********************************/

/* Thread */
typedef struct thread {
  int id;
  pthread_t pthread;
  struct thread_pool *my_thread_pool;
} thread;

/* Threadpool */
typedef struct thread_pool {
  thread **threads;
  volatile int num_threads_alive;
  pthread_mutex_t count_mutex;
  my_queue_t jobqueue;
  int jobqueue_len;
  my_semaphore *exist_jobs;
  pthread_mutex_t rwmutex;
} thread_pool;

thread_pool *my_thread_pool;
static void *thread_do(struct thread *work_thread) {
  printf("create new thread %d\n", work_thread->id + 1);
  /* Assure all threads have been created before starting serving */
  thread_pool *my_thread_pool = work_thread->my_thread_pool;

  /* inited done*/
  pthread_mutex_lock(&my_thread_pool->count_mutex);
  my_thread_pool->num_threads_alive += 1;
  pthread_mutex_unlock(&my_thread_pool->count_mutex);

  while (1) {
    /* wait for job */
    semaphore_wait(my_thread_pool->exist_jobs);
    /* has job, get it from jobqueue and execute it */
    void (*func_buff)(int);
    int arg_buff;

    pthread_mutex_lock(&my_thread_pool->rwmutex);
    if (my_thread_pool->jobqueue_len > 0) {
      /* get it from jobqueue */
      my_item_t *job_p = my_thread_pool->jobqueue.head;
      DL_DELETE(my_thread_pool->jobqueue.head, my_thread_pool->jobqueue.head);
      my_thread_pool->jobqueue_len--;
      /* it jobqueue not empty, notify other threads to execute it */
      if (my_thread_pool->jobqueue_len > 0)
        semaphore_post(my_thread_pool->exist_jobs);
      pthread_mutex_unlock(&my_thread_pool->rwmutex);
      if (job_p) {
        func_buff = job_p->function;
        arg_buff = job_p->arg;
        /* execute it */
        func_buff(arg_buff);
        /* free it, avoid memory leak */
        free(job_p);
      }
    } else {
      pthread_mutex_unlock(&my_thread_pool->rwmutex);
    }
  }
  pthread_mutex_lock(&my_thread_pool->count_mutex);
  my_thread_pool->num_threads_alive--;
  pthread_mutex_unlock(&my_thread_pool->count_mutex);
  printf("thread exit\n");

  return NULL;
}

void async_init(int num_threads) {
  if (num_threads < 0) {
    num_threads = 0;
  }
  my_thread_pool = (struct thread_pool *)malloc(sizeof(struct thread_pool));

  my_thread_pool->num_threads_alive = 0;
  my_thread_pool->threads =
      (struct thread **)malloc(num_threads * sizeof(struct thread *));

  my_thread_pool->jobqueue_len = 0;
  pthread_mutex_init(&(my_thread_pool->count_mutex), NULL);
  my_thread_pool->exist_jobs = (struct my_semaphore *)malloc(sizeof(struct my_semaphore));
  pthread_mutex_init(&(my_thread_pool->rwmutex), NULL);
  semaphore_init(my_thread_pool->exist_jobs, 0);

  /* Thread init */
  int n;
  for (n = 0; n < num_threads; n++) {
    struct thread **work_thread = &my_thread_pool->threads[n];
    *work_thread = (struct thread *)malloc(sizeof(struct thread));
    (*work_thread)->my_thread_pool = my_thread_pool;
    (*work_thread)->id = n;
    pthread_create(&(*work_thread)->pthread, NULL, (void *(*)(void *))thread_do,
                   (*work_thread));
    pthread_detach((*work_thread)->pthread);
  }

  /* Wait for threads to initialize */
  while (my_thread_pool->num_threads_alive != num_threads) {
  }
}

void async_run(void (*hanlder)(int), int args) {
  /* create new job */
  my_item_t *new_job = (my_item_t *)malloc(sizeof(my_item_t));
  new_job->function = hanlder;
  new_job->arg = args;
  pthread_mutex_lock(&my_thread_pool->rwmutex);
  /* add to jobqueue and notify work thread to execute it */
  my_thread_pool->jobqueue_len++;
  DL_APPEND(my_thread_pool->jobqueue.head, new_job);
  semaphore_post(my_thread_pool->exist_jobs);
  pthread_mutex_unlock(&my_thread_pool->rwmutex);
}