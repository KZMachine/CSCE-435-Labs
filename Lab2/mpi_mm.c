/******************************************************************************
* FILE: mpi_mm.c
* DESCRIPTION:  
*   MPI Matrix Multiply - C Version
*   In this code, the master task distributes a matrix multiply
*   operation to numtasks-1 worker tasks.
*   NOTE:  C and Fortran versions of this code differ because of the way
*   arrays are stored/passed.  C arrays are row-major order but Fortran
*   arrays are column-major order.
* AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
*   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
* LAST REVISED: 09/29/2021
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int main (int argc, char *argv[])
{
    
int sizeOfMatrix;
if (argc == 2)
{
    sizeOfMatrix = atoi(argv[1]);
}
else
{
    printf("\n Please provide the size of the matrix");
    return 0;
}
int	numtasks,              /* number of tasks in partition */
	taskid,                /* a task identifier */
	numworkers,            /* number of worker tasks */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	rows,                  /* rows of matrix A sent to each worker */
	averow, extra, offset, /* used to determine rows sent to each worker */
	i, j, k, rc;           /* misc */
double	a[sizeOfMatrix][sizeOfMatrix],           /* matrix A to be multiplied */
	b[sizeOfMatrix][sizeOfMatrix],           /* matrix B to be multiplied */
	c[sizeOfMatrix][sizeOfMatrix];           /* result matrix C */
MPI_Status status;

double start_time_recv, end_time_recv, start_time_calc, end_time_calc, start_time_send, end_time_send;
double final_time_max_recv, final_time_min_recv, final_time_max_send, final_time_min_send, final_time_max_calc, final_time_min_calc;
double total = 0.0;
double final_time_initial, final_time_sendrcv, final_time = 0.0;
double start_time = 0.0;

MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
if (numtasks < 2 ) {
  printf("Need at least two MPI tasks. Quitting...\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  exit(1);
  }
numworkers = numtasks-1;
double saved_times[numworkers];
MPI_Barrier(MPI_COMM_WORLD);
/**************************** master task ************************************/

   if (taskid == MASTER)
   {
      start_time = MPI_Wtime();
      printf("Start Time 1: %f", start_time);
      double start_time_initial = MPI_Wtime();
      // INITIALIZATION PART FOR THE MASTER PROCESS STARTS HERE
      
      printf("mpi_mm has started with %d tasks.\n",numtasks);
      printf("Initializing arrays...\n");
      for (i=0; i<sizeOfMatrix; i++)
         for (j=0; j<sizeOfMatrix; j++)
            a[i][j]= i+j;
      for (i=0; i<sizeOfMatrix; i++)
         for (j=0; j<sizeOfMatrix; j++)
            b[i][j]= i*j;
            
      //INITIALIZATION PART FOR THE MASTER PROCESS ENDS HERE
      double end_time_initial = MPI_Wtime();
      final_time_initial = end_time_initial - start_time_initial;
      
      //SEND-RECEIVE PART FOR THE MASTER PROCESS STARTS HERE
      double start_time_sendrcv = MPI_Wtime();
      /* Send matrix data to the worker tasks */
      averow = sizeOfMatrix/numworkers;
      extra = sizeOfMatrix%numworkers;
      offset = 0;
      mtype = FROM_MASTER;
      for (dest=1; dest<=numworkers; dest++)
      {
         rows = (dest <= extra) ? averow+1 : averow;   	
         printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
         MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&a[offset][0], rows*sizeOfMatrix, MPI_DOUBLE, dest, mtype,
                   MPI_COMM_WORLD);
         MPI_Send(&b, sizeOfMatrix*sizeOfMatrix, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
         offset = offset + rows;
      }

      /* Receive results from worker tasks */
      mtype = FROM_WORKER;
      for (i=1; i<=numworkers; i++)
      {
         source = i;
         MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&c[offset][0], rows*sizeOfMatrix, MPI_DOUBLE, source, mtype, 
                  MPI_COMM_WORLD, &status);
         printf("Received results from task %d\n",source);
      }
      
      //SEND-RECEIVE PART FOR THE MASTER PROCESS ENDS HERE
        double end_time_sendrcv = MPI_Wtime();
        final_time_sendrcv = end_time_sendrcv - start_time_sendrcv;

      /* Print results - you can uncomment the following lines to print the result matrix */
      /*
      printf("******************************************************\n");
      printf("Result Matrix:\n");
      for (i=0; i<sizeOfMatrix; i++)
      {
         printf("\n"); 
         for (j=0; j<sizeOfMatrix; j++) 
            printf("%6.2f   ", c[i][j]);
      }
      printf("\n******************************************************\n");
      printf ("Done.\n");
      */
      
      
   }


/**************************** worker task ************************************/
   
   
   if (taskid > MASTER)
   {
      //RECEIVING PART FOR WORKER PROCESS STARTS HERE
      
      start_time_recv = MPI_Wtime();

      mtype = FROM_MASTER;
      MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&a, rows*sizeOfMatrix, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&b, sizeOfMatrix*sizeOfMatrix, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
      
      end_time_recv = MPI_Wtime();
      final_time_max_recv = final_time_min_recv = end_time_recv - start_time_recv;
      // printf("Final Time: %f\n", final_time);

      // printf("Receive time for task %d: %f\n", taskid, final_time_min_recv);

      //RECEIVING PART FOR WORKER PROCESS ENDS HERE
      
      
      //CALCULATION PART FOR WORKER PROCESS STARTS HERE
      
      start_time_calc = MPI_Wtime();

      for (k=0; k<sizeOfMatrix; k++)
         for (i=0; i<rows; i++)
         {
            c[i][k] = 0.0;
            for (j=0; j<sizeOfMatrix; j++)
               c[i][k] = c[i][k] + a[i][j] * b[j][k];
         }
         
      end_time_calc = MPI_Wtime();
      final_time_max_calc = final_time_min_calc = end_time_calc - start_time_calc;
      // printf("Calc time for task %d: %f\n", taskid, final_time_max_calc);

      //CALCULATION PART FOR WORKER PROCESS ENDS HERE
      
      
      //SENDING PART FOR WORKER PROCESS STARTS HERE
      
      start_time_send = MPI_Wtime();

      mtype = FROM_WORKER;
      MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&c, rows*sizeOfMatrix, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);

      end_time_send = MPI_Wtime();

      final_time_max_send = final_time_min_send = end_time_send - start_time_send;
      
      //SENDING PART FOR WORKER PROCESS ENDS HERE
   }
   double end_time = 0.0;
   //USE MPI_Reduce here to calculate the minimum, maximum and the average times for the worker processes.
   double total_sum_recv, max_time_recv, min_time_recv = 0;
   double total_sum_calc, max_time_calc, min_time_calc = 0;
   double total_sum_send, max_time_send, min_time_send = 0;
   if (taskid == MASTER)
   {
      // end_time = MPI_Wtime();
      // printf("Start Time 2: %f", start_time);
      // final_time = end_time - start_time;
      final_time_max_recv = final_time_max_calc = final_time_max_send = 0.0;
    //   final_time_min_recv = final_time_min_calc = final_time_min_send = __DBL_MAX__;
   //  printf("Whole computation: %f\n", final_time);
   //  printf("initialization time: %f\n", final_time_initial);
   //  printf("Send/Receive time: %f\n", final_time_sendrcv);
   }

   MPI_Reduce(&final_time_min_recv, &min_time_recv, 1, MPI_DOUBLE, MPI_MIN, MASTER, MPI_COMM_WORLD);
   MPI_Reduce(&final_time_max_recv, &total_sum_recv, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
   MPI_Reduce(&final_time_max_recv, &max_time_recv, 1, MPI_DOUBLE, MPI_MAX, MASTER, MPI_COMM_WORLD);

   MPI_Reduce(&final_time_min_calc, &min_time_calc, 1, MPI_DOUBLE, MPI_MIN, MASTER, MPI_COMM_WORLD);
   MPI_Reduce(&final_time_max_calc, &total_sum_calc, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
   MPI_Reduce(&final_time_max_calc, &max_time_calc, 1, MPI_DOUBLE, MPI_MAX, MASTER, MPI_COMM_WORLD);

   MPI_Reduce(&final_time_min_send, &min_time_send, 1, MPI_DOUBLE, MPI_MIN, MASTER, MPI_COMM_WORLD);
   MPI_Reduce(&final_time_max_send, &total_sum_send, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
   MPI_Reduce(&final_time_max_send, &max_time_send, 1, MPI_DOUBLE, MPI_MAX, MASTER, MPI_COMM_WORLD);
   MPI_Barrier(MPI_COMM_WORLD);
   end_time = MPI_Wtime();
   MPI_Finalize();
   if (taskid == MASTER)
   {
      final_time = end_time - start_time;
      // final_time_max_recv = final_time_max_calc = final_time_max_send = 0.0;
    //   final_time_min_recv = final_time_min_calc = final_time_min_send = __DBL_MAX__;
    printf("Whole computation: %f\n", final_time);
    printf("initialization time: %f\n", final_time_initial);
    printf("Send/Receive time: %f\n", final_time_sendrcv);
   }
   /*if (final_time)

   if(total_sum_recv != 0.0)
   {
      printf("Average recv time: %f\n", (double)total_sum_recv/numworkers);
   }*/
   if(max_time_recv != 0.0)
   {
      printf("Max recv time: %f\n", max_time_recv);
   }
   /*
   if(min_time_recv != 0.0)
   {
      printf("Min recv time: %f\n", min_time_recv);
   }

   if(total_sum_calc != 0.0)
   {
      printf("Average calc time: %f\n", (double)total_sum_calc/numworkers);
   }
   if(max_time_calc != 0.0)
   {
      printf("Max calc time: %f\n", max_time_calc);
   }
   if(min_time_calc != 0.0)
   {
      printf("Min calc time: %f\n", min_time_calc);
   }

   if(total_sum_send != 0.0)
   {
      printf("Average send time: %f\n", (double)total_sum_send/numworkers);
   }
   if(max_time_send != 0.0)
   {
      printf("Max send time: %f\n", max_time_send);
   }
   if(min_time_send != 0.0)
   {
      printf("Min send time: %f\n", min_time_send);
   }*/
}