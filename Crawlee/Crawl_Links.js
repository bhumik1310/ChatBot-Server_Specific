import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const sanitizeFilename = require('sanitize-filename');
const path = require('path');
const url = require('url');
import { JSDOM } from 'jsdom';
import { Readability } from '@mozilla/readability';
const tldextract = require('tld-extract') 




import { Dataset,CheerioCrawler } from 'crawlee';
const fs = require('fs');
const csv = require('csv-parser');

const crawler = new CheerioCrawler({
    async requestHandler({ $,request,body }) {
        const title = $('title').text();

        function cleanString(input) {

            return input.replace(/<header[^>]*>[\s\S]*?<\/header>/gi, '')
           .replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, '')
           .replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, '')
           .replace(/<header-class[^>]*>[\s\S]*?<\/header-class>/gi, '')
           .replace(/<footer-class[^>]*>[\s\S]*?<\/footer-class>/gi, '')
           .replace(/<main-menu[^>]*>[\s\S]*?<\/main-menu>/gi, '')
           .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
           .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
           .replace(/<[^>]*>/g, '')
           .replace(/\s+/g, ' ');
         
        }
        var dom = new JSDOM(body);

        // Create a Readability object and parse the DOM document
        var reader = new Readability(dom.window.document);
        var article = reader.parse();    
        var content = article.textContent;
        console.log(`The title: ${title}`);
        await Dataset.pushData({
            url:request.url,
            content : content
            // page_content: cleanString(body)
        })
    }
})



const csvFilePath = './storage/key_value_stores/my-data/OUTPUT.csv';

const columnValues = [];

fs.createReadStream(csvFilePath)
  .pipe(csv())
  .on('data', (row) => {
    const value = row[Object.keys(row)[0]];
    columnValues.push(value);
  })
  .on('end', () => {
    // console.log('Column values:', columnValues);
    // console.log('Type of Thing', typeof columnValues);
    // console.log('At index 0 (before): \n', columnValues[0])
    // // columnValues.splice(0,1);
    // console.log('At index 0 (after): \n', columnValues[0])

  })
  .on('error', (error) => {
    console.error('Error reading CSV:', error);
  });


// Start the crawler with the provided URLs
await crawler.run(columnValues);
// Crawler saves files temporarily as json in datasets



// Check if the folder exists



var arrayOfFiles=[]
try {
    arrayOfFiles = fs.readdirSync("./storage/datasets/default")
    // console.log(arrayOfFiles)
  } catch(e) {
    console.log(e)
  }
  // Iterating over each file in a folder
  for (const element of arrayOfFiles){
    var filePath="./storage/datasets/default/"+element;
    // Read the JSON file
    fs.readFile(filePath, 'utf8', (err, data) => {
            if (err) {
        console.error('Error reading the JSON file:', err);
        return;
    }

    try {
        // Parse the JSON data
        const jsonData = JSON.parse(data);

        // Create a string with the data you want to append
        let appendData = '\n';
        for (const key in jsonData) {
            if (jsonData.hasOwnProperty(key)) {
                const value = jsonData[key];
                appendData += `${key}: ${value}\n`;
            }
        }
        
        // console.log(typeof appendData);
        const txt_filename = jsonData['url'] + '.txt';
        // console.log(txt_filename)
        const parsedUrl = new URL(jsonData['url']);
        const domain = parsedUrl.hostname
        domain.replace(/.+\/\/|www.|\..+/g,'');
        const folderPath = domain
        const totalPath = "./CrawledData/"+folderPath
        //Check if the url has been crawled already or not
        if (!fs.existsSync(totalPath)) {
          // If it doesn't exist, create the folder
          fs.mkdirSync(totalPath);
          console.log(`Folder '${folderPath}' created.`);
        } else { 
          console.log(`Folder '${folderPath}' already exists.`);
        }
        const filename = sanitizeFilename(txt_filename)
        const filePath = path.join(totalPath,filename);
        fs.writeFile(filePath, appendData, (err) => {
          if (err) {
            console.error('Error writing the file:', err);
          } else {
            console.log(`File '${filename}' has been written to '${folderPath}'.`);
          }
        });
        // console.log(sanitizeFilename(txt_filename));
    
    } catch (parseError) {
        console.error('Error parsing the JSON data:', parseError);
    }
});
  }
