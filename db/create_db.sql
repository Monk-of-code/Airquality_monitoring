SELECT  format('CREATE DATABASE %I','airquality')
WHERE NOT EXISTS(SELECT 1 FROM pg_database WHERE datname = 'airquality')
\gexec
