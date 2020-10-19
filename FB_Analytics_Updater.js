function updateFBAnalytics() {
  // Gets information from Facebook and updates spreadsheet
  const week_in_seconds = 604800;

  var insightsRaw = UrlFetchApp.fetch("https://graph.facebook.com/v7.0/PropertyImprovementEnterprises/insights?metric=page_fans,page_post_engagements,page_impressions_unique,page_views_total&access_token=REMOVED");
  var insights = JSON.parse(insightsRaw.getContentText());
  
  var end_time = new Date/1000; //Current date in seconds
  var start_time = end_time - week_in_seconds;
 
  var postRaw = UrlFetchApp.fetch("https://graph.facebook.com/v7.0/PropertyImprovementEnterprises?fields=published_posts.summary(total_count).since(" + start_time + ").until(" + end_time + ")&access_token=REMOVED");
  var post = JSON.parse(postRaw.getContentText());
  
  var sheet = SpreadsheetApp.getActiveSheet();
  var currentRow = sheet.getLastRow() + 1; //Uses row below latest
  sheet.getRange(currentRow,2).setValue(insights.data[0].values[1].end_time.substring(0,10)); //Date
  sheet.getRange(currentRow,3).setValue(insights.data[0].values[1].value); //Like count
  sheet.getRange(currentRow,4).setValue(insights.data[0].values[1].value - sheet.getRange(lastRow, 3).getValue()); //Like change
  sheet.getRange(currentRow,5).setValue(post.published_posts.summary.total_count); //Number of posts
  sheet.getRange(currentRow,6).setValue(insights.data[4].values[1].value); //Engagement (likes, comments, shares)
  sheet.getRange(currentRow,7).setValue(insights.data[5].values[1].value); //Unique impressions (Unique people seeing content: posts, page, etc.)
  sheet.getRange(currentRow,8).setValue(insights.data[6].values[1].value); //Page visits
  
}