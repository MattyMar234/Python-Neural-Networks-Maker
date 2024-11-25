import os
import threading
import psycopg2
from typing import Optional, List, Tuple, Any

import psycopg2.pool
from Database.DatabaseConnectionParametre import DatabaseParametre
from Database.Tables import TableBase
import Globals


# class PostgresDB:
#     """Gestore delle connessioni a PostgreSQL per utilizzo in singolo processo."""

#     _lock = threading.Lock()  # Lock globale per sincronizzazione

#     def __init__(self, parametre: DatabaseParametre):
#         assert isinstance(parametre, DatabaseParametre), "parametre deve essere un'istanza di DatabaseParametre"

#         self._connection_parametre = parametre
#         self._connection: Optional[psycopg2.extensions.connection] = None

#         Globals.APP_LOGGER.info(f"Inizializzazione connessione PostgresDB: {parametre.Host}:{parametre.Port}")
#         self._connect()

#     def _connect(self):
#         """Inizializza una singola connessione al database."""
#         try:
#             self._connection = psycopg2.connect(
#                 host=self._connection_parametre.Host,
#                 database=self._connection_parametre.DatabaseName,
#                 user=self._connection_parametre.User,
#                 password=self._connection_parametre.Password,
#                 port=self._connection_parametre.Port,
#                 connect_timeout=self._connection_parametre.Timeout
#             )
#             Globals.APP_LOGGER.info(f"Connessione creata per {self._connection_parametre.Host}:{self._connection_parametre.Port}")
#         except Exception as e:
#             Globals.APP_LOGGER.error(f"Errore nella connessione al database: {e}")
#             raise e

#     def is_connected(self) -> bool:
#         """Verifica se la connessione è attiva."""
#         return self._connection is not None and not self._connection.closed

#     def execute_query(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> None:
#         """Esegue una query senza restituire risultati."""
#         if not self.is_connected():
#             raise Exception("La connessione al database è inattiva.")

#         try:
#             with self._connection.cursor() as cur:
#                 # cur.execute("SET tcp_keepalives_idle = 30;")
#                 # cur.execute("SET tcp_keepalives_interval = 30;")
#                 # cur.execute("SET tcp_keepalives_count = 30;")
                
#                 cur.execute(query, params)
#                 self._connection.commit()
#         except Exception as e:
#             self._connection.rollback()
#             Globals.APP_LOGGER.error(f"Errore durante l'esecuzione della query: {e}")
#             raise e

#     def fetch_results(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> List[Tuple[Any, ...]]:
#         """Esegue una query e restituisce i risultati."""
#         # if not self.is_connected():
#         #     raise Exception("La connessione al database è inattiva.")

#         try:
#             with self._connection.cursor() as cur:
#                 # cur.execute("SET tcp_keepalives_idle = 30;")
#                 # cur.execute("SET tcp_keepalives_interval = 30;")
#                 # cur.execute("SET tcp_keepalives_count = 30;")
                
#                 cur.execute(query, params)
#                 results = cur.fetchall()
#                 return results
#         except psycopg2.ProgrammingError as e:
#             if str(e) == "no results to fetch":
#                 return []
#             Globals.APP_LOGGER.error(f"Errore durante il recupero dei risultati: {e}")
#             raise
#         except Exception as e:
#             Globals.APP_LOGGER.error(f"Errore generale durante il recupero dei risultati: {e}")
#             raise

#     def close_connection(self):
#         """Chiude la connessione al database."""
#         if self._connection and not self._connection.closed:
#             self._connection.close()
#             Globals.APP_LOGGER.info("Connessione al database chiusa.")
#         else:
#             Globals.APP_LOGGER.warning("Connessione già chiusa o inattiva.")

class PostgresDB:

    __instances: dict = {}
    __lock = threading.Lock()  # Lock per gestire l'accesso concorrente
    __instance_count = 0
    __SINGLE_MODE = False
    
    def __new__(cls, parametre: DatabaseParametre, *args, **kwargs):
        
        assert isinstance(parametre, DatabaseParametre), "parametre is not a DatabaseParametre"
        key = (parametre.Host, parametre.Port)  # Chiave unica per host e porta
        
        with cls.__lock:
            if key not in cls.__instances:
                instance = super(PostgresDB, cls).__new__(cls)
                cls.__instances[key] = instance
                
                if parametre is not None:
                    Globals.APP_LOGGER.info(f"New PostgresDB connection on {parametre.Host}:{parametre.Port}. PID: {os.getpid()}, {instance}, count: {PostgresDB.__instance_count}")
            return cls.__instances[key]
    
    
    def __init__ (self, parametre: DatabaseParametre | None, mainProcess: bool = False) -> None:
        assert type(parametre) == DatabaseParametre, "parametre is not a DatabaseParametre"
        
        with PostgresDB.__lock:
            PostgresDB.__instance_count += 1
            #print(f"New PostgresDB connection. PID: {os.getpid()}, {self}, count: {PostgresDB.__instance_count}")
        
        self._conenectionParametre: DatabaseParametre  | None = parametre
        self._mainProcess: bool = mainProcess
        self._connection = None
        
        #Globals.APP_LOGGER.info(f"New PostgresDB connection on {parametre.Host}:{parametre.Port}")
        print((f"New PostgresDB connection on {parametre.Host}:{parametre.Port}"))
        
        # if self._mainProcess:
        #     self._logger = logging.getLogger("database")
        #     self._logger.info(f"{'-'*40}New Session{'-'*40}")
        #     self._logger.debug(f"parametre: {self._conenectionParametre}")
        
        
        #self.__setup()
        if self._conenectionParametre is not None:
            self._connect()
        

    # def __del__(self):
        
    #     with PostgresDB.__lock:
    #         PostgresDB.__instance_count -= 1
    #         #print(f"End PostgresDB connection. PID: {os.getpid()}, {self}, count: {PostgresDB.__instance_count}")
            
    #         if PostgresDB.__instance_count == 0:
    #             self.close_pool()
        
    
    def _connect(self):
        
        if not PostgresDB.__SINGLE_MODE:
            try:
                self._conn_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn  = 1,
                    maxconn  = 100,#self._conenectionParametre.MaxConn*2,
                    host     = self._conenectionParametre.Host,
                    database = self._conenectionParametre.DatabaseName,
                    user     = self._conenectionParametre.User,
                    password = self._conenectionParametre.Password,
                    port     = self._conenectionParametre.Port,
                    connect_timeout = self._conenectionParametre.Timeout
                )
                
                if self._conn_pool == None:
                    Globals.APP_LOGGER.error(f"Connection pool is None for procces: {self.__name}")
                    raise Exception("Connection pool is None")
                
                # if self._conn_pool:
                #     if self._mainProcess:
                #         self._logger.info(f"Connection pool created successfully for instance: {self.__name}")
                #     else:
                #         print(f"Connection pool created successfully for instance: {self.__name}")
                    
            except Exception as e:
                Globals.APP_LOGGER.error(f"Error creating connection pool: {e}")
                os._exit(0)
                
        else:
            self._connection = psycopg2.connect(
                host     = self._conenectionParametre.Host,
                database = self._conenectionParametre.DatabaseName,
                user     = self._conenectionParametre.User,
                password = self._conenectionParametre.Password,
                port     = self._conenectionParametre.Port,
                connect_timeout = self._conenectionParametre.Timeout
            )
        
    def is_connected(self) -> bool:
        return self._conn_pool is not None

    


    def __checkPool(self):
        if self._conn_pool is None:
            #self._logger.error("Connection pool is None. Call 'connect()' first.")
            raise Exception("Connection pool is None. Call 'connect()' first.")

    def createTable(self, table: TableBase) -> None:
        self.execute_query(table.createTable_Query())

    def deleteTable(self, table: TableBase) -> None:
        self.execute_query(table.dropTableQuery())

    def execute_query(self, query:str, params=None) -> None:

        if PostgresDB.__SINGLE_MODE:
            try:
                with self._connection.cursor() as cur:
                    cur.execute(f"SET statement_timeout TO {self._conenectionParametre.Timeout * 1000};")  # Timeout in millisecondi
                    cur.execute(query, params)
                    self._connection.commit()

            except Exception as e:
                Globals.APP_LOGGER.error(f"Error creating cursor: {e}")
                return None
        else:
            self.__checkPool()
            #self._logger.debug(f"execute_query params: query:{query}, params:{params}")

            conn = self._conn_pool.getconn()  # Ottieni una connessione dal pool
            #startTime = time.time()

            try:
                with conn.cursor() as cur:
                    #cur.execute(f"SET statement_timeout TO {self._conenectionParametre.Timeout * 1000};")  # Timeout in millisecondi
                    cur.execute(query, params)
                    conn.commit()
                    
            except Exception as e:
                conn.rollback()
                Globals.APP_LOGGER.error(f"Error during query execution: {e}")
            finally:
                #dTime = time.time() - startTime
                #self._logger.debug(f"Query eseguita in {dTime} s.")
                self._conn_pool.putconn(conn)  # Rilascia la connessione al pool

    def fetch_results(self, query, params=None):
         
        if PostgresDB.__SINGLE_MODE:
            try:
                with conn.cursor() as cur:
                    #cur.execute(f"SET statement_timeout TO {self._conenectionParametre.Timeout * 1000};")  # Timeout in millisecondi
                    cur.execute(query, params)
                    results = cur.fetchall()
                    return results
            except Exception as e:
                
                if str(e) == "no results to fetch":
                    return None
                
                Globals.APP_LOGGER.error(f"error during result fetching: {e}")
                raise e
        else:
            #with self.__lock:  # Assicura l'accesso esclusivo alla connessione
            self.__checkPool()
            conn = self._conn_pool.getconn()
            try:
                with conn.cursor() as cur:
                    #cur.execute(f"SET statement_timeout TO {self._conenectionParametre.Timeout * 1000};")  # Timeout in millisecondi
                    cur.execute(query, params)
                    results = cur.fetchall()
                    return results
            except Exception as e:
                
                if str(e) == "no results to fetch":
                    return None
                
                Globals.APP_LOGGER.error(f"error during result fetching: {e}")
                raise e
            
            finally:
                # Assicurati sempre di rilasciare la connessione al pool
                self._conn_pool.putconn(conn, close=False)

    def close_pool(self):
        if self._conn_pool:
            self._conn_pool.closeall()
            self._conn_pool = None
            Globals.APP_LOGGER.info("Connection pool closed.")
        else:
            Globals.APP_LOGGER.warning("Connection pool is already closed.")
