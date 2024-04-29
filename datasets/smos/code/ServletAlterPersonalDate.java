package smos.application.userManagement;

import java.io.IOException;
import java.sql.SQLException;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import smos.Environment;
import smos.bean.User;
import smos.exception.DuplicatedEntityException;
import smos.exception.EntityNotFoundException;
import smos.exception.InvalidValueException;
import smos.exception.MandatoryFieldException;
import smos.storage.ManagerUser;
import smos.storage.connectionManagement.exception.ConnectionException;
/**
 * Change server Password of the logged-in user.
 * 
 * @author Napolitano Vincenzo.
 * 
 */
public class ServletAlterPersonalDate extends HttpServlet {

	private static final long serialVersionUID = 1L;
	/**
	 * Definition of the doGet method
	 * 
	 * @param pRequest
	 * @param pResponse
	 * 
	 */
	protected void doGet(HttpServletRequest pRequest, 
			HttpServletResponse pResponse) {
		String gotoPage = "showUserList";
		String errorMessage = "";
		HttpSession session = pRequest.getSession();
		ManagerUser managerUser = ManagerUser.getInstance();
		User user = (User) session.getAttribute("loggedUser");
		
		
		User loggedUser = (User) session.getAttribute("loggedUser");
		
		try {
			if (loggedUser == null) {
				pResponse.sendRedirect("./index.htm");
				return;
			}
			if (!managerUser.isAdministrator(loggedUser)) {
				errorMessage =  "The logged-in User does not have access to" +
						"functionality'!";
				gotoPage = "./error.jsp";
			}
			
	
			//verify that the login is unique.
			if (managerUser.existsLogin(user))
				throw new InvalidValueException("La login inserita esiste gia'. Inserire una nuova login.");
			user.setPassword(pRequest.getParameter("pass"));
			//We update
			if (!managerUser.exists(user)){
				managerUser.update(user);
			}else {
				int userId = managerUser.getUserId(user);
				if (user.getId()==userId)
					managerUser.update(user);
				else 
					throw new DuplicatedEntityException("Existing user");
			}
			
		} catch (SQLException SQLException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + SQLException.getMessage();
			gotoPage = "./error.jsp";
			SQLException.printStackTrace();
		} catch (ConnectionException connectionException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + connectionException.getMessage();
			gotoPage = "./error.jsp";
			connectionException.printStackTrace();
		}  catch (EntityNotFoundException entityNotFoundException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + entityNotFoundException.getMessage();
			gotoPage = "./error.jsp";
			entityNotFoundException.printStackTrace();
		} catch (MandatoryFieldException mandatoryFieldException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + mandatoryFieldException.getMessage();
			gotoPage = "./error.jsp";
			mandatoryFieldException.printStackTrace();
		} catch (InvalidValueException invalidValueException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + invalidValueException.getMessage();
			gotoPage = "./error.jsp";
			invalidValueException.printStackTrace();
		} catch (DuplicatedEntityException duplicatedEntityException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + duplicatedEntityException.getMessage();
			gotoPage = "./error.jsp";
			duplicatedEntityException.printStackTrace();
		} catch (IOException ioException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE + ioException.getMessage();
			gotoPage = "./error.jsp";
			ioException.printStackTrace();
		}
			
		session.setAttribute("errorMessage", errorMessage);
		try {
			pResponse.sendRedirect(gotoPage);
		} catch (IOException ioException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE + ioException.getMessage();
			gotoPage = "./error.jsp";
			ioException.printStackTrace();
		}
		
	}

	/**
	 * Definition of the doPost method
	 * 
	 * @param pRequest
	 * @param pResponse
	 * 
	 */
	protected void doPost(HttpServletRequest pRequest, 
			HttpServletResponse pResponse) {
		this.doGet(pRequest, pResponse);
	}
	
}

