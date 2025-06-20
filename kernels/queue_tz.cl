#include "barrier.h"

int tz_enqueue_block(__global volatile tz_queue_t * t, uint32_t newnode){
    while(1){
        uint32_t te = VREAD(t->tail);
        uint32_t ate = te;
        uint32_t tt = VREAD(t->nodes[ate]);
        //The next slot of the tail
        uint32_t temp = (ate + 1) % MY_QUEUE_LENGTH;
        //Find the actual tail
        while(tt != NULL_0 && tt != NULL_1){
            //check consistency
            if(te != VREAD(t->tail)) break;
            //if tail meets head, may be full
            if(temp == VREAD(t->head)) break;
            //now check the next cell
            tt = VREAD(t->nodes[temp]);
            ate = temp;
            temp = (ate + 1) % MY_QUEUE_LENGTH;
        }
        if(tt != NULL_0 && tt != NULL_1)continue;
        //check the tail's consistency
        if(te != VREAD(t->tail)) continue;
        //check if queue is full
        if(temp == VREAD(t->head)){
            ate = (temp + 1) % MY_QUEUE_LENGTH;
            tt = VREAD(t->nodes[ate]);
            //the cell after head is OCCUPIED
            if(tt != NULL_0 && tt != NULL_1){
                continue; //queue full
            }
            //if head rewind try update null
            if(!ate)
                VWRITE(t->vnull,tt);
            //help the dequeue to update head
            VOLATILE_CAS(t->head, temp, ate);
            //try enqueue again
            continue;
        }
        //check tail consistency
        if(te != VREAD(t->tail)) continue;
        //get the actual tail and try enqueue
        if(VOLATILE_CAS(t->nodes[ate], tt, newnode) == tt){
            if(temp%2==0)// enqueue has succeeded
                VOLATILE_CAS(t->tail, te, temp);
            return 0;
        }
    }
}

int tz_dequeue_block(__global volatile tz_queue_t *t, volatile uint32_t * oldnode){
    do{
        uint32_t th = VREAD(t->head); // read the head
        //here is the one we want to dequeue
        uint32_t temp = (th + 1) % MY_QUEUE_LENGTH;
        uint32_t tt = VREAD(t->nodes[temp]);
        //find the actual head after this loop
        while (tt == NULL_0 || tt == NULL_1){
            //check the head's consistency
           if(th != VREAD(t->head)) break;
           //two consecutive NULL means EMPTY return
           if(temp == VREAD(t->tail)) break;
           temp = (temp + 1) % MY_QUEUE_LENGTH; // next cell
           tt = VREAD(t->nodes[temp]);
        }
        if (tt == NULL_0 || tt == NULL_1)continue;
        //check the head's consistency
        if(th != VREAD(t->head)) continue;
        //check whether the Queue is empty
        if(temp == VREAD(t->tail)){
            //help the enqueue to update end
            VOLATILE_CAS(t->tail, temp, (temp + 1) % MY_QUEUE_LENGTH);
            continue; //try dequeue again
        }
        //if dequeue rewind to 0
        //switching NULL to avoid ABA
        uint32_t tnull;
        if(temp){
            if(temp<th){
                tnull = VREAD(t->nodes[0]);
            }else{
                tnull = VREAD(t->vnull);
            }
        }else{
            tnull = VREAD(t->vnull) ^ 1; //TODO:WTF is this?
        }
        //check the head's consistency
        if (th != VREAD(t->head)) continue;
        //get the actual head, null means empty
        if(VOLATILE_CAS(t->nodes[temp], tt, tnull) == tt){
            //if dequeue rewind to 0
            //switch NULLs to avoid ABA
            if(!temp) VWRITE(t->vnull, tnull);
            if(temp%2 == 0) VOLATILE_CAS(t->head, th, temp);
            *oldnode = tt;
            return 0;
        }
    }while(1);
}

int tz_enqueue(__global volatile tz_queue_t * t, uint32_t newnode){
    while(1){
        uint32_t te = VREAD(t->tail);
        uint32_t ate = te;
        uint32_t tt = VREAD(t->nodes[ate]);
        //The next slot of the tail
        uint32_t temp = (ate + 1) % MY_QUEUE_LENGTH;
        //Find the actual tail
        while(tt != NULL_0 && tt != NULL_1){
            //check consistency
            if(te != VREAD(t->tail)) break;
            //if tail meets head, may be full
            if(temp == VREAD(t->head)) break;
            //now check the next cell
            tt = VREAD(t->nodes[temp]);
            ate = temp;
            temp = (ate + 1) % MY_QUEUE_LENGTH;
        }
        if(tt != NULL_0 && tt != NULL_1)continue;
        //check the tail's consistency
        if(te != VREAD(t->tail)) continue;
        //check if queue is full
        if(temp == VREAD(t->head)){
            ate = (temp + 1) % MY_QUEUE_LENGTH;
            tt = VREAD(t->nodes[ate]);
            //the cell after head is OCCUPIED
            if(tt != NULL_0 && tt != NULL_1)
                return 1; //queue full
            //if head rewind try update null
            if(!ate)
                VWRITE(t->vnull,tt);
            //help the dequeue to update head
            VOLATILE_CAS(t->head, temp, ate);
            //try enqueue again
            continue;
        }
        //check tail consistency
        if(te != VREAD(t->tail)) continue;
        //get the actual tail and try enqueue
        if(VOLATILE_CAS(t->nodes[ate], tt, newnode) == tt){
            if(temp%2==0)// enqueue has succeeded
                VOLATILE_CAS(t->tail, te, temp);
            return 0;
        }
    }
}

int tz_dequeue(__global volatile tz_queue_t *t, volatile uint32_t * oldnode){
    do{
        uint32_t th = VREAD(t->head); // read the head
        //here is the one we want to dequeue
        uint32_t temp = (th + 1) % MY_QUEUE_LENGTH;
        uint32_t tt = VREAD(t->nodes[temp]);
        //find the actual head after this loop
        while (tt == NULL_0 || tt == NULL_1){
            //check the head's consistency
           if(th != VREAD(t->head)) break;
           //two consecutive NULL means EMPTY return
           if(temp == VREAD(t->tail)) return 1;
           temp = (temp + 1) % MY_QUEUE_LENGTH; // next cell
           tt = VREAD(t->nodes[temp]);
        }
        if (tt == NULL_0 || tt == NULL_1)continue;
        //check the head's consistency
        if(th != VREAD(t->head)) continue;
        //check whether the Queue is empty
        if(temp == VREAD(t->tail)){
            //help the enqueue to update end
            VOLATILE_CAS(t->tail, temp, (temp + 1) % MY_QUEUE_LENGTH);
            continue; //try dequeue again
        }
        //if dequeue rewind to 0
        //switching NULL to avoid ABA
        uint32_t tnull;
        if(temp){
            if(temp<th){
                tnull = VREAD(t->nodes[0]);
            }else{
                tnull = VREAD(t->vnull);
            }
        }else{
            tnull = VREAD(t->vnull) ^ 1; //TODO:WTF is this?
        }
        //check the head's consistency
        if (th != VREAD(t->head)) continue;
        //get the actual head, null means empty
        if(VOLATILE_CAS(t->nodes[temp], tt, tnull) == tt){
            //if dequeue rewind to 0
            //switch NULLs to avoid ABA
            if(!temp) VWRITE(t->vnull, tnull);
            if(temp%2 == 0) VOLATILE_CAS(t->head, th, temp);
            *oldnode = tt;
            return 0;
        }
    }while(1);
}

