
import os
import traceback
from typing import Dict
import psycopg2
import logging
from psycopg2 import pool
from threading import Lock
import time

from Database.DatabaseConnectionParametre import DatabaseParametre
from Database.Tables import *









#conda install anaconda::psycopg2

import threading

class PostgresDB:

    __instances: dict = {}
    __lock = threading.Lock()  # Lock per gestire l'accesso concorrente
    __instance_count = 0
    
    #singleton
    def __new__(cls, parametre: DatabaseParametre, *args, **kwargs):
        
        assert isinstance(parametre, DatabaseParametre), "parametre is not a DatabaseParametre"
        key = (parametre.Host, parametre.Port)  # Chiave unica per host e porta
        
        with cls.__lock:
            if key not in cls.__instances:
                instance = super(PostgresDB, cls).__new__(cls)
                cls.__instances[key] = instance
                print(f"New PostgresDB connection. PID: {os.getpid()}, {instance}, count: {PostgresDB.__instance_count}")
            return cls.__instances[key]
    
    
    def __init__ (self, parametre: DatabaseParametre, mainProcess: bool = False) -> None:
        assert type(parametre) == DatabaseParametre, "parametre is not a DatabaseParametre"
        
        with PostgresDB.__lock:
            PostgresDB.__instance_count += 1
            #print(f"New PostgresDB connection. PID: {os.getpid()}, {self}, count: {PostgresDB.__instance_count}")
        
        self._conenectionParametre: DatabaseParametre = parametre
        self._mainProcess: bool = mainProcess
        
        if self._mainProcess:
            self._logger = logging.getLogger("database")
            self._logger.info(f"{'-'*40}New Session{'-'*40}")
            self._logger.debug(f"parametre: {self._conenectionParametre}")
        
        
        #self.__setup()
        self.__connect()
        

    def __del__(self):
        
        with PostgresDB.__lock:
            PostgresDB.__instance_count -= 1
            #print(f"End PostgresDB connection. PID: {os.getpid()}, {self}, count: {PostgresDB.__instance_count}")
            
            if PostgresDB.__instance_count == 0:
                self.close_pool()
        
    
    def __connect(self):
        
        self.__name = os.getpid()
        
        if self._mainProcess:
            self._logger.info(f"Creating connection pool for instance: {self.__name}")
        
        try:
            self._conn_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn  = 1,
                maxconn  = self._conenectionParametre.MaxConn,
                host     = self._conenectionParametre.Host,
                database = self._conenectionParametre.DatabaseName,
                user     = self._conenectionParametre.User,
                password = self._conenectionParametre.Password,
                port     = self._conenectionParametre.Port,
                connect_timeout = self._conenectionParametre.Timeout
            )
            
            if self._conn_pool == None:
                print(f"Connection pool is None for procces: {self.__name}")
                raise Exception("Connection pool is None")
            
            # if self._conn_pool:
            #     if self._mainProcess:
            #         self._logger.info(f"Connection pool created successfully for instance: {self.__name}")
            #     else:
            #         print(f"Connection pool created successfully for instance: {self.__name}")
                
        except Exception as e:
            
            if self._mainProcess:
                self._logger.error(f"Instance: {self.__name}, Error creating connection pool: {e}")
            else:
                print(f"Instance: {self.__name}, Error creating connection pool: {e}")
            
            # traceback.print_exc()
            # tb = traceback.extract_tb(e.__traceback__)
            
            # for frame in tb:
            #     print(f"Errore nel file '{frame.filename}', linea {frame.lineno}, nella funzione '{frame.name}'")
            #     print(f"Linea di codice: {frame.line}")
        
            self._conn_pool = None

    def is_connected(self) -> bool:
        return self._conn_pool is not None

    


    def __checkPool(self):
        if self._conn_pool is None:
            #self._logger.error("Connection pool is None. Call 'connect()' first.")
            raise Exception("Connection pool is None. Call 'connect()' first.")

    def createTable(self, table: TableBase) -> None:
        self.execute_query(table.createTableQuery())

    def deleteTable(self, table: TableBase) -> None:
        self.execute_query(table.dropTableQuery())

    def execute_query(self, query, params=None) -> None:

        self.__checkPool()
        self._logger.debug(f"execute_query params: query:{query}, params:{params}")

        conn = self._conn_pool.getconn()  # Ottieni una connessione dal pool
        #startTime = time.time()

        try:
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout TO {self._conenectionParametre.Timeout * 1000};")  # Timeout in millisecondi
                cur.execute(query, params)
                conn.commit()
                
        except Exception as e:
            if self._mainProcess:
                self._logger.error(f"Error during query execution: {e}")
            conn.rollback()
        finally:
            #dTime = time.time() - startTime
            #self._logger.debug(f"Query eseguita in {dTime} s.")
            self._conn_pool.putconn(conn)  # Rilascia la connessione al pool

    def fetch_results(self, query, params=None):
        
        #with self.__lock:  # Assicura l'accesso esclusivo alla connessione
        self.__checkPool()
        conn = self._conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout TO {self._conenectionParametre.Timeout * 1000};")  # Timeout in millisecondi
                cur.execute(query, params)
                results = cur.fetchall()
                return results
        except Exception as e:
            #self._logger.error(f"error during result fetching: {e}")
            return None
        
        finally:
            self._conn_pool.putconn(conn)

    def close_pool(self):
        if self._conn_pool:
            self._conn_pool.closeall()
            self._conn_pool = None
            #self._logger.info("Connection pool closed.")
        else:
            #self._logger.warning("Connection pool is already closed.")
            pass