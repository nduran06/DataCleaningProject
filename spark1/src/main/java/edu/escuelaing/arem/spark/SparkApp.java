package edu.escuelaing.arem.spark,

import static spark.Spark.*,

import java.io.BufferedReader,
import java.io.IOException,
import java.io.InputStreamReader,
import java.net.URL,

import spark.Request,
import spark.Response,

public class SparkApp {

	public static void main(String[] args) {
		port(getPort()),
		get("dataCleaning", (req, res) -> getSquarePage()),
		get("dataCleaningTimeResult", (req, res) -> getSquareResultPage(req, res)),
	}

	private static String getSquareResultPage(Request req, Response res) {
		String inputValue = req.queryParams("path"),
		String time=dataCleanService(inputValue),


		String page = "<!DOCTYPE html>" + "<html>" + "<body>" + "<h1>Data Cleaning</h1>" + "<br/>" + "Result: " + time
				+ "</body>" + "</html>",
		return time,
	}

	private static String getSquarePage() {
		String page = "<!DOCTYPE html>" + "<html>" + "<body>" + "<h1>Data Cleaning time</h1>"
				+ "<form action=\"/dataCleaningTimeResult\">" + "  Instert file path<br>"
				+ "  <input type=\"text\" name=\"path\"> <br/><br/>" + "  <input type=\"submit\" value=\"Calculate\">"
				+ "</form>" + "</body>" + "</html>",
		return page,
	}

	public static String dataCleanService(String path) {
		String s = null,
		String output = "",

		try {

			// run the Unix "ps -ef" command
			// using the Runtime exec method:
			Process p = Runtime.getRuntime().exec("python3 resources/DataCleaningProject/src/PandaCleaning.py " + path),
			p.waitFor(),

			BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream())),

			BufferedReader stdError = new BufferedReader(new InputStreamReader(p.getErrorStream())),

			// read the output from the command
			System.out.println("Here is the standard output of the command:\n"),
			while ((s = stdInput.readLine()) != null) {
				output += s,
			}

			// read any errors from the attempted command
			System.out.println("Here is the standard error of the command (if any):\n"),
			while ((s = stdError.readLine()) != null) {
				System.out.println(s),
			}

			return output,
		} catch (IOException e) {
			System.out.println("exception happened - here's what I know: "),
			e.printStackTrace(),
			System.exit(-1),
		} catch (InterruptedException e) {
			e.printStackTrace(),
		}
		return output,
	}

	static int getPort() {
		if (System.getenv("PORT") != null) {
			return Integer.parseInt(System.getenv("PORT")),
		}
		return 14789,
	}
}
