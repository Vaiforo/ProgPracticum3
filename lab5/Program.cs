using System;
using System.Collections;
using Python.Runtime;

namespace PythonCSharpIntegration
{
    class Program
    {
        static void Main(string[] args)
        {
            AppDomain.CurrentDomain.SetData("APP_CONTEXT_DELETE_AUTOMATIC_CLEANUP", true);

            Runtime.PythonDLL = @"C:\Program Files\Python312\python312.dll";
            PythonEngine.Initialize();

            try
            {
                using (Py.GIL())
                {
                    dynamic sys = Py.Import("sys");

                    string scriptPath = Directory.GetCurrentDirectory();
                    sys.path.append(scriptPath);

                    string jsonFilePath = Path.Combine(scriptPath, "data.json");

                    dynamic pyScript = Py.Import("lab5");

                    PyObject result = pyScript.main("data.json");

                    var resultDynamic = result.AsManagedObject(typeof(object));

                    Console.WriteLine("Тип результата: " + resultDynamic.GetType().FullName);
                    Console.WriteLine("Результат от Python:");
                    Console.WriteLine(resultDynamic?.ToString() ?? "null");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Произошла ошибка:");
                Console.WriteLine(ex.Message);
                Console.WriteLine(ex.StackTrace);
            }
            finally
            {
                PythonEngine.Shutdown();
            }
        }
    }
}
