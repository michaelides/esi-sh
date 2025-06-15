// www/js/cookie_handler.js

/**
 * Sets a cookie in the browser.
 * @param {string} name - The name of the cookie.
 * @param {string} value - The value of the cookie.
 * @param {number} [days] - Optional number of days until the cookie expires. If not provided, it's a session cookie.
 */
function setCookie(name, value, days) {
    var expires = "";
    if (days) {
        var date = new Date();
        date.setTime(date.getTime() + (days*24*60*60*1000));
        expires = "; expires=" + date.toUTCString();
    }
    document.cookie = name + "=" + (value || "")  + expires + "; path=/; SameSite=Lax"; // Added SameSite=Lax
    console.log("Cookie operation: setCookie called for " + name + " with value (first 50 chars): " + String(value).substring(0,50));
}

/**
 * Gets a cookie value by name.
 * @param {string} name - The name of the cookie.
 * @returns {string|null} The cookie value or null if not found.
 */
function getCookie(name) {
    var nameEQ = name + "=";
    var ca = document.cookie.split(';');
    for(var i=0;i < ca.length;i++) {
        var c = ca[i];
        while (c.charAt(0)==' ') c = c.substring(1,c.length);
        if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length,c.length);
    }
    return null;
}

/**
 * Deletes a cookie by name.
 * @param {string} name - The name of the cookie to delete.
 */
function deleteCookie(name) {
    document.cookie = name +'=; Path=/; Expires=Thu, 01 Jan 1970 00:00:01 GMT; SameSite=Lax;'; // Added SameSite=Lax
    console.log("Cookie operation: deleteCookie called for " + name);
}

/**
 * Event listener for when Shiny connects.
 * Reads 'user_id' and 'ltm_preference' cookies and sends them to Shiny inputs
 * `cookie_user_id_loaded` and `cookie_ltm_preference_loaded`.
 */
$(document).on('shiny:connected', function(event) {
    console.log("Shiny connected. Attempting to read cookies for initial state.");
    var userId = getCookie("user_id");
    var ltmPref = getCookie("ltm_preference");

    if (userId) {
        Shiny.setInputValue("cookie_user_id_loaded", userId, {priority: "event"});
        // console.log("Found user_id cookie: " + userId); // Reduced verbosity
    } else {
        Shiny.setInputValue("cookie_user_id_loaded", "None", {priority: "event"});
        // console.log("user_id cookie not found.");
    }

    if (ltmPref) {
        Shiny.setInputValue("cookie_ltm_preference_loaded", ltmPref, {priority: "event"});
        // console.log("Found ltm_preference cookie: " + ltmPref);
    } else {
        Shiny.setInputValue("cookie_ltm_preference_loaded", "None", {priority: "event"});
        // console.log("ltm_preference cookie not found.");
    }
    console.log("Finished sending initial cookie values to Shiny (if they existed).");
});

/**
 * Custom message handler for Shiny to set a cookie.
 * Expects a message object with `name`, `value`, and optionally `days`.
 */
Shiny.addCustomMessageHandler("setCookie", function(message) {
    setCookie(message.name, message.value, message.days);
});

/**
 * Custom message handler for Shiny to delete a cookie.
 * Expects a message object with `name`.
 */
Shiny.addCustomMessageHandler("deleteCookie", function(message) {
    deleteCookie(message.name);
});

/**
 * Custom message handler for Shiny to copy text to the clipboard.
 * Expects a message object with `text`.
 * Sends back "clipboard_copy_success" or "clipboard_copy_error" to Shiny.
 */
Shiny.addCustomMessageHandler("copyToClipboard", function(message) {
    // console.log("JS: copyToClipboard handler received text (first 50 chars): ", message.text.substring(0, 50)); // Debug
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
        navigator.clipboard.writeText(message.text)
            .then(function() {
                // console.log("JS: Text successfully copied to clipboard."); // Debug
                Shiny.setInputValue("clipboard_copy_success", { timestamp: Date.now() }, {priority: "event"});
            })
            .catch(function(err) {
                console.error("JS: Failed to copy text: ", err);
                Shiny.setInputValue("clipboard_copy_error", { error: err.toString(), timestamp: Date.now() }, {priority: "event"});
            });
    } else {
        console.error("JS: Clipboard API not available.");
        Shiny.setInputValue("clipboard_copy_error", { error: "Clipboard API not available", timestamp: Date.now() }, {priority: "event"});
    }
});
