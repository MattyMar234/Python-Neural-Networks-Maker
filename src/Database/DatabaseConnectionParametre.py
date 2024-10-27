


class DatabaseParametre(object):
    def __init__(self, host: str, database: str, user: str, password: str, port: int, maxconn: int, timeout: int):
        self._host = host
        self._database = database
        self._user = user
        self._password = password
        self._port = port
        self._maxconn = maxconn
        self.timeout = timeout
    
    @property 
    def Host(self) -> str:
        return self._host
    
    @property
    def DatabaseName(self) -> str:
        return self._database
    
    @property
    def User(self) -> str:
        return self._user
    
    @property
    def Password(self) -> str:
        return self._password
    
    @property
    def Port(self) -> int:
        return self._port
    
    @property
    def MaxConn(self) -> int:
        return self._maxconn
    
    @property
    def Timeout(self) -> int:
        return self.timeout
    

    def __str__(self) -> str:
        return f"DatabaseParametre(name={self._host}, type={self._database}, user={self._user}, password={self._password}, port={self._port}, maxconn={self._maxconn}, timeout={self.timeout})"