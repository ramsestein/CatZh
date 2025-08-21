@echo off
setlocal enabledelayedexpansion

:: Configuración
set SAMPLES=10000
set MAX_ITERATIONS=1000
set ITERATION=1

echo ========================================
echo PIPELINE AUTOMATICO CA-ZH EN BUCLE
echo ========================================
echo.
echo Configuracion:
echo - Muestras por iteracion: %SAMPLES%
echo - Numero de iteraciones: %MAX_ITERATIONS%
echo.
echo Presiona CTRL+C para detener manualmente
echo ========================================
echo.

:: Verificar que existen los archivos Python
if not exist csv_muestra.py (
    echo ERROR: No se encuentra csv_muestra.py
    pause
    exit /b 1
)

if not exist p1_embeddings.py (
    echo ERROR: No se encuentra p1_embeddings.py
    pause
    exit /b 1
)

:: Activar entorno virtual
echo ========================================
echo ACTIVANDO ENTORNO CATZH
echo ========================================
call catzh\Scripts\activate
if errorlevel 1 (
    echo ERROR: No se pudo activar el entorno catzh
    pause
    exit /b 1
)
echo Entorno catzh activado correctamente
echo.

:LOOP
echo.
echo ========================================
echo ITERACION %ITERATION% de %MAX_ITERATIONS%
echo ========================================
echo Hora inicio: %date% %time%
echo.

:: PASO 1: Extraer muestras
echo [PASO 1/2] Extrayendo %SAMPLES% muestras aleatorias...
echo ----------------------------------------
python csv_muestra.py --samples %SAMPLES%

if errorlevel 1 (
    echo ERROR: Fallo en csv_muestra.py
    goto :ERROR_EXIT
)

:: Buscar el archivo CSV más reciente generado
for /f "delims=" %%i in ('dir /b /od muestra_traducciones_*.csv 2^>nul') do set LATEST_CSV=%%i

if not defined LATEST_CSV (
    echo ERROR: No se genero archivo CSV de muestras
    goto :ERROR_EXIT
)

:: PASO 2: Procesar con embeddings (modo automático)
echo.
echo [PASO 2/2] Procesando con embeddings (modo automatico)...
echo ----------------------------------------
python p1_embeddings.py "!LATEST_CSV!" --no-interactive

if errorlevel 1 (
    echo ERROR: Fallo en p1_embeddings.py
    goto :ERROR_EXIT
)

:: Incrementar contador
set /a ITERATION+=1

:: Verificar si llegamos al máximo
if !ITERATION! GTR %MAX_ITERATIONS% (
    echo.
    echo ========================================
    echo COMPLETADAS TODAS LAS ITERACIONES
    echo ========================================
    goto :SUCCESS_EXIT
)

:: Pequeña pausa para no saturar
echo.
echo Esperando 5 segundos antes de la siguiente iteracion...
timeout /t 5 /nobreak >nul

:: Continuar el bucle
goto :LOOP

:SUCCESS_EXIT
echo.
echo ========================================
echo PROCESO COMPLETADO EXITOSAMENTE
echo ========================================
echo Total de iteraciones completadas: %MAX_ITERATIONS%
echo Total de muestras procesadas: %MAX_ITERATIONS% x %SAMPLES% = !TOTAL_SAMPLES!
set /a TOTAL_SAMPLES=%MAX_ITERATIONS%*%SAMPLES%
echo.
echo Revisa los resultados en:
echo - Archivo Excel acumulado de muestras limpias
echo - Carpeta p1_results con todas las metricas
echo ========================================
pause
exit /b 0

:ERROR_EXIT
echo.
echo ========================================
echo PROCESO DETENIDO POR ERROR
echo ========================================
echo Se produjo un error durante la iteracion %ITERATION%
echo Revisa los mensajes anteriores para mas detalles.
echo ========================================
pause
exit /b 1